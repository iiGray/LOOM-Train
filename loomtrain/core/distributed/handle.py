from collections import deque
from collections.abc import Iterator
from uuid import uuid4

from typing import TypeVar, TYPE_CHECKING, Callable
from enum import Enum, auto
import os, ray, functools
from ray.util.placement_group import PlacementGroup

if TYPE_CHECKING:
    from loomtrain.core.distributed.resource import (
        RayResourcePool, RayHandleMixin
    )

RayHandle = TypeVar("RayHandle", bound = "RayHandleMixin")

RayHandleRemote = TypeVar("RayHandleRemoteFuncOrProperty")

RAY_BROADCAST_DEFAULT = False

def sort_placement_group_by_node_ip(pgs: list[PlacementGroup], with_index: bool = False) -> list[PlacementGroup]:
    """
    From Verl

    Sort the placement groups by node ip, all bundles in a single placement group should be on the same node.

    FSDPCheckpointManager saves sharded model states and optimizer states in local storage, which requires RANK
    to be consistent across nodes when resume from checkpoint.

    With this function, if there's only one resource pool and there's no node change, RANK should be consistent
    across nodes in multiple ray jobs, even if the whole ray cluster is restarted.
    """
    node_ip = {node["NodeID"]: node["NodeManagerAddress"] for node in ray.nodes()}
    pg_ip = {}
    for pg in pgs:
        specs = ray._private.state.state.placement_group_table(pg.id)
        # all bunles should be on the same node
        node_id = specs["bundles_to_node_id"][0]
        pg_ip[pg.id] = node_ip[node_id]
    
    if with_index:
        indexed_pgs = [(i, pg) for i, pg in enumerate(pgs)]
        return sorted(indexed_pgs, key = lambda x: (pg_ip[x[1].id], x[0]))
    return sorted(pgs, key=lambda pg: pg_ip[pg.id])



class RayRole(Enum):
    DATA = auto()
    ACTOR = auto()
    ROLLOUT = auto()
    REFERENCE = auto()
    REWARD = auto()
    CRITIC = auto()


class _RayIterator(Iterator):
    def __init__(self, worker, stream_id, prefetch = 2):
        self._worker = worker
        self._stream_id = stream_id
        self._prefetch = prefetch
        self._pending = deque()
        self._started = False
        self._closed = False

    def __getstate__(self):
        if self._started or self._closed:
            raise TypeError(
                "An iterator can only be transferred before consumption."
            )
        return self._worker, self._stream_id, self._prefetch

    def __setstate__(self, state):
        self.__init__(*state)

    def __iter__(self):
        return self

    def _request_next(self):
        return self._worker._next_iterator_item.remote(self._stream_id)

    def __next__(self):
        if self._closed: raise StopIteration

        try:
            if not self._started:
                self._started = True
                for _ in range(self._prefetch):
                    self._pending.append(self._request_next())
            finished, value = ray.get(self._pending.popleft())

            if finished:
                self.close()
                raise StopIteration

            self._pending.append(self._request_next())
            return value

        except BaseException:
            self.close()
            raise

    def close(self):
        if self._closed: return

        self._closed = True
        self._pending.clear()
        try:
            self._worker._close_iterator.remote(self._stream_id)
        except Exception: pass


@ray.remote
class RayWorker:
    def __init__(
        self, 
        lazy_init_object: RayHandle, 
        resource_pool: "RayResourcePool",
        bin_pack: bool = True,
        detached: bool = False,
        sharing_with: RayHandle = None):
        from loomtrain.core.arguments import set_args
        set_args(vars(lazy_init_object).pop("_worker_args_"))


        self._iterators = {}
        
        self.instance = lazy_init_object._ray_lazy_initialize_(
            resource_pool, 
            bin_pack,
            detached,
            sharing_with,
            is_remote=True)

    def ready(self): return True

    def _prepare_strategy(self):
        return self.instance._prepare_strategy()

    def _get_property(self, prop_name):
        return getattr(self.instance, prop_name)

    def _execute_method(self, method_name, *args, **kwargs):
        input_iterators = [
            value for value in (*args, *kwargs.values())
            if isinstance(value, _RayIterator)
        ]
        try:
            method = getattr(self.instance, method_name)
            result = method(*args, **kwargs)

            if isinstance(result, Iterator):
                stream_id = uuid4().hex

                exported = _RayIterator(
                    worker = ray.get_runtime_context().current_actor,
                    stream_id = stream_id
                )

                self._iterators[stream_id] = (result, input_iterators)

                input_iterators = []

                return exported
            return result

        finally:
            for iterator in input_iterators:
                iterator.close()


    def _next_iterator_item(self, stream_id):
        state = self._iterators.get(stream_id)

        if state is None: return True, None

        iterator, _ = state

        try:
            return False, next(iterator)

        except StopIteration:
            self._close_iterator(stream_id)
            return True, None

        except BaseException:
            self._close_iterator(stream_id)
            raise

    def _close_iterator(self, stream_id):
        state = self._iterators.pop(stream_id, None)

        if state is None: return

        iterator, inputs = state

        try:
            close = getattr(iterator, "close", None)
            if close is not None:
                close()
        finally:
            for value in inputs:
                value.close()




    def get_node_id(self):
        return ray.get_runtime_context().get_node_id()

    def get_cuda_visible_devices(self):
        return os.environ.get("CUDA_VISIBLE_DEVICES", "")


class RayHandleMeta(type):
    """Each class that inherit this meta class will be able to automatically initilize a remote group containing many workers."""
    def __call__(cls: "type[RayHandleMixin]", *args, **kwargs):
        obj = cls.__new__(cls)
        obj._stored_init_args_ = args
        obj._stored_init_kwargs_ = kwargs
        
        obj._is_ray_proxy = False
        obj._ray_workers = []
        obj._is_ray_lazy_initialized = False

        def perform_init(self: "RayHandleMixin", resource_pool: "RayResourcePool", bin_pack: bool = True, detached: bool = False, sharing_with: list[RayHandle] = None, is_remote=False):
            if getattr(self, "_is_ray_lazy_initialized", False):
                return self    

            if is_remote:
                cls.__init__(self, *self._stored_init_args_, **self._stored_init_kwargs_)
                del self._stored_init_args_
                del self._stored_init_kwargs_
                self._is_ray_lazy_initialized = True
                return self
            
            from loomtrain.core.distributed.resource import RayResourcePool
            assert isinstance(resource_pool, RayResourcePool), f"resource pool must be a RayResourcePool, not `{type(resource_pool)}`"

            strategy = "STRICT_PACK" if bin_pack else "PACK"
            pgs = resource_pool.get_placement_groups(strategy = strategy)

            rank = -1

            for pg_idx, (pg_pos_idx, pg) in enumerate(sort_placement_group_by_node_ip(pgs, with_index = True)):
                local_world_size = resource_pool.nprocs[pg_pos_idx]
                assert local_world_size <= pg.bundle_count
                
                if pg_idx == 0:
                    master_addr, master_port = resource_pool._get_master_addr_port(pg, bundle_index = 0)
                
                for local_rank in range(local_world_size):
                    rank += 1

                    current_sharing = sharing_with[rank] if sharing_with is not None else None

                    self._ray_workers.append(
                        resource_pool._create_worker(
                            master_addr = master_addr,
                            master_port = master_port,
                            name = f"{cls.__name__}_{id(self)}",
                            rank = rank,
                            local_rank = local_rank,
                            local_world_size = local_world_size,
                            pg_idx = pg_idx,
                            pg = pg,
                            detached = detached,
                            sharing_with = current_sharing,
                        ).remote(self, resource_pool, bin_pack, detached, current_sharing)
                    )

            self._is_ray_proxy = True
            self._is_ray_lazy_initialized = True
            return self

        obj._ray_lazy_initialize_ = perform_init.__get__(obj)
        return obj


def _auto_reduce(elements, num_workers: int):
    if not hasattr(elements, "__len__"):
        result = [elements for _ in range(num_workers)]
    elif len(elements) == 0:
        result = [[] for _ in range(num_workers)]
    elif len(elements) == num_workers:
        return elements
    elif len(elements) % num_workers == 0:
        split_len = len(elements) // num_workers
        result = [elements[i: i + split_len] for i in range(0, len(elements), split_len)]
    elif num_workers % len(elements) == 0:
        repeat_len = num_workers // len(elements)
        result = [elements[i: i + 1] for i in range(len(elements)) for _ in range(repeat_len)]
    else:
        raise RuntimeError(f"the length of the input argument must be divisible by the number of workers.")

    return result



class RayHandleOption:
    def __init__(self, handle: "RayHandleMixin", reduce_input: bool = False, sync_output: bool = False, execute: Callable = None, gather: Callable = None, get_first: bool = False):
        self._handle__ = handle
        self._reduce_input__ = reduce_input
        self._sync_output__ = sync_output
        self._execute__ = execute
        self._gather__ = gather
        self._get_first__ = get_first


    def _execute_broadcast(self, workers: list[RayWorker], method_name: str, args: tuple, kwargs: dict, is_property: bool = False):
        if self._execute__ is not None:
            futures = self._execute__(workers, method_name, args, kwargs, is_property) 
        elif is_property:
            futures = [w._get_property.remote(method_name) for w in workers]
        elif self._reduce_input__: 
            reduced_args = tuple(zip(*[_auto_reduce(a, len(workers)) for a in args]))
            if len(reduced_args) == 0: reduced_args = [()] * len(workers)
            reduced_v = {k: _auto_reduce(v, len(workers)) for k, v in kwargs.items()}
            reduced_kwargs = [{k: v[i] for k, v in reduced_v.items()} for i in range(len(workers))]
            futures = [w._execute_method.remote(method_name, *reduced_args[i], **reduced_kwargs[i]) for i, w in enumerate(workers)]
        else:
            futures = [w._execute_method.remote(method_name, *args, **kwargs) for w in workers]
        
        if self._get_first__:
            assert self._gather__ is None
            return ray.get(next(iter(futures)))

        if self._sync_output__ or self._gather__: futures = ray.get(futures)
        if self._gather__: futures = self._gather__(futures)
        return futures
    


    def __getattribute__(self, name):
        state = object.__getattribute__(self, "__dict__")
        if "_handle__" not in state or (
            name.startswith("__") and name.endswith("__")
        ):
            return super().__getattribute__(name)

        if name in [
            "_handle__", "_reduce_input__", "_sync_output__", "_execute__", "_gather__",
            "_execute_broadcast", "_get_first__"
        ]: return super().__getattribute__(name)

        attr = getattr(type(self._handle__), name, None)

        if isinstance(attr, property):
            if not getattr(attr.fget, "_is_ray_broadcast_method", RAY_BROADCAST_DEFAULT):
                raise AttributeError(f"'{name}' is not a valid ray broadcast property.")
            return self._execute_broadcast(self._handle__._ray_workers, name, args=(), kwargs={}, is_property=True)

        if not (callable(attr) and getattr(attr, "_is_ray_broadcast_method", RAY_BROADCAST_DEFAULT)):
            raise AttributeError(f"'{name}' is not a valid ray broadcast method.")    

        def wrapper(*args, **kwargs):
            return self._execute_broadcast(self._handle__._ray_workers, name, args, kwargs)
        return wrapper

REMOTE_NAME = "dist"

class RayHandleMixin(metaclass = RayHandleMeta):
    def _ray_initialize(self, resource_pool: "RayResourcePool", bin_pack: bool = True, detached: bool = False, sharing_with: list[RayHandle] = None, is_remote: bool = False):
        '''Lazy initialize, Has been implemented in the meta class. Be intialized in `trainer.fit`'''
        return self._ray_lazy_initialize_(
            resource_pool, bin_pack, detached, sharing_with, is_remote
        )

    def ready(self):
        return ray.get([
            worker.ready.remote()
            for worker in self._ray_workers
        ])
    
    def prepare_strategy(self):
        [worker._prepare_strategy.remote() for worker in self._ray_workers]

    def dist(self: "RayHandle", reduce_input: "bool" = False, sync_output: "bool" = False, execute: "Callable" = None, gather: "Callable" = None, get_first: bool = False) -> "RayHandle":
        return RayHandleOption(self, reduce_input, sync_output, execute, gather, get_first)

    def __getattribute__(self, name):
        state = object.__getattribute__(self, "__dict__")
        if not state.get("_is_ray_proxy", False):
            return super().__getattribute__(name)
        if not super().__getattribute__("_is_ray_proxy"): return super().__getattribute__(name)
        if name in ["_is_ray_proxy", "_ray_workers", "_ray_lazy_initialize_", "_is_ray_lazy_initialized", "_stored_init_args_", "_stored_init_kwargs_"]:
            return super().__getattribute__(name)

        class_attr = getattr(type(self), name, None)

        if isinstance(class_attr, property) and getattr(class_attr.fget, "_is_ray_broadcast_method", RAY_BROADCAST_DEFAULT):
            raise RuntimeError(
                    f"You MUST call `.{REMOTE_NAME}(...)` before accessing the broadcast property `{name}`. "
                    f"Example: obj.{REMOTE_NAME}().{name}"
            )
        if callable(class_attr) and getattr(class_attr, "_is_ray_broadcast_method", RAY_BROADCAST_DEFAULT):
            raise RuntimeError(
                    f"You MUST call `.{REMOTE_NAME}(...)` before calling the broadcast method `{name}()`. "
                    f"Example: obj.{REMOTE_NAME}(reduce_input=True).{name}(...)"
            )

        return super().__getattribute__(name)



    @staticmethod
    def remote(func: "RayHandleRemote") -> "RayHandleRemote":
        """Delare that a func could not be called locally."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result
        wrapper._is_ray_broadcast_method = True
        
        return wrapper

    @staticmethod
    def local(func: "RayHandleRemote") -> "RayHandleRemote":
        """Delare that a func could not be called locally."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            return result
        wrapper._is_ray_broadcast_method = False
        
        return wrapper