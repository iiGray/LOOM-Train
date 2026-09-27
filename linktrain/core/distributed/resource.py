from typing import Optional, TypeVar, TYPE_CHECKING
import os, ray, socket, logging
from contextlib import ExitStack
from ray.util.placement_group import PlacementGroup, placement_group
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy, PlacementGroupSchedulingStrategy
from linktrain.core.arguments import args
from linktrain.core.distributed.handle import (
    RayRole,
    RayWorker,
    RayHandleMixin,
    sort_placement_group_by_node_ip
)


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))



LOCAL_RANK_NAME = "RAY_LOCAL_RANK"
LOCAL_WORLD_SIZE_NAME = "RAY_LOCAL_WORLD_SIZE"




RayHandle = TypeVar("RayHandle", bound = RayHandleMixin)

@ray.remote
def get_master_addr_port(master_port_range: Optional[list[int]] = None, exclude_ports: list = []) -> tuple[str, str]:
    addr = ray.util.get_node_ip_address().strip("[]")

    excluded = (
        {int(port) for port in exclude_ports}
        if exclude_ports is not None
        else set()
    )

    if master_port_range is None:
        with ExitStack() as stack:
            while True:
                s = stack.enter_context(socket.socket())
                s.bind(("", 0))
                port = s.getsockname()[1]
                if port not in excluded:
                    break
    else:
        port = master_port_range[0]
        while port < master_port_range[1]:
            if port in excluded:
                port += 1
                continue
            try:
                with socket.socket() as s:
                    s.bind(("", port))
                    break
            except OSError:
                port += 1  # Increment port number if already in use
                logger.info("Port %d is already in use, trying port %d", port - 1, port)
        else:
            raise RuntimeError(f"Could not find a free port in range {master_port_range}")
    return addr, str(port)




class RayResourcePool:
    def __init__(self, pid: int, nprocs: list[int], ncpus_per_proc: int, ngpus_per_proc: int, roles: list[RayRole]):
        self.pid = pid

        self.nprocs = nprocs
        self.ncpus_per_proc = ncpus_per_proc
        self.ngpus_per_proc = ngpus_per_proc
        self.roles = roles

        self.pgs = None

        self._master_addr = None
        self._master_port = None
        self._exclude_ports = []

        self._anchor_workers = None

        self._worker_group_count = 0


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    @property
    def world_size(self):
        return sum(self.nprocs)

    @property
    def name(self) -> str:
        return "_".join(map(str, self.roles))

    def get_placement_groups(self, strategy = "STRICT_PACK"):
        if self.pgs is not None: return self.pgs
        bundle = dict(CPU = self.ncpus_per_proc)
        if self.ngpus_per_proc > 0:
            bundle["GPU"] = self.ngpus_per_proc

        pgs = [
            placement_group(
                bundles = [bundle.copy() for _ in range(nproc)],
                strategy = strategy,
                name = f"RayResourcePool_{self.pid}_proc_{idx}",
                lifetime = None,
            )
            for idx, nproc in enumerate(self.nprocs)
        ]

        ray.get([pg.ready() for pg in pgs])
        self.pgs = pgs
        return pgs
    
    def _get_master_addr_port(self, pg: PlacementGroup, bundle_index: int = 0, master_port_range = None):
        """From Ver. Get master addr and port for this worker group"""
        _master_addr, _master_port = ray.get(
            get_master_addr_port.options(
                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group = pg, placement_group_bundle_index = bundle_index
                ),
                num_cpus = 0,
            ).remote(master_port_range = master_port_range, exclude_ports = self._exclude_ports),
        )
        if self._master_addr is None:
            self._master_addr = _master_addr
            self._master_port = _master_port
        else:
            assert self._master_addr == _master_addr
            logger.debug(f"{_master_addr=} {_master_port=}")
        
        self._exclude_ports.append(_master_port)
        return _master_addr, _master_port


    def _create_worker(self, master_addr, master_port,  name: str, rank: int, local_rank: int, local_world_size: int, pg_idx: int, pg: PlacementGroup, detached: bool, sharing_with: RayWorker = None) -> RayWorker:

        env_vars = {
            "WORLD_SIZE": str(self.world_size),
            "RANK": str(rank),
            "WG_PREFIX": f"{self.name}_p{self.pid}_g{self._worker_group_count}",
            "WG_BACKEND": "ray",
            f"{LOCAL_WORLD_SIZE_NAME}": str(local_world_size),
            f"{LOCAL_RANK_NAME}": str(local_rank),
            "LOCAL_WORLD_SIZE": str(local_world_size),
            "LOCAL_RANK": "0", # for device
            
            "MASTER_ADDR": master_addr,
            "MASTER_PORT": master_port,
        }


        ray_options = dict(
            runtime_env = dict(env_vars = env_vars),
            name = f"{self.name}_{name}_{pg_idx}:{local_rank}"
        )

        if detached:
            ray_options.update(dict(lifetime = "detached"))
        
        if sharing_with is None:
            ray_options.update(
                dict(
                    scheduling_strategy = PlacementGroupSchedulingStrategy(
                        placement_group = pg,
                        placement_group_bundle_index = local_rank,
                    )
                )
            )
            if self.ngpus_per_proc:
                ray_options.update(dict(num_gpus = self.ngpus_per_proc)) 

        else:
            target_node_id = ray.get(sharing_with.get_node_id.remote())
            cuda_visible_devices = ray.get(sharing_with.get_cuda_visible_devices.remote())
            ray_options.update(
                dict(
                    scheduling_strategy = NodeAffinitySchedulingStrategy(
                        node_id = target_node_id, 
                        soft = False
                    ),
                    num_cpus = 0,
                    num_gpus = 0
                )
            )
            ray_options['runtime_env']['env_vars'].update(
                dict(
                    CUDA_VISIBLE_DEVICES = cuda_visible_devices,
                    RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES = "1",
                )
            ) #### 和 verl不一样，verl是传参数
        

        return RayWorker.options(** ray_options)
    

    def init_worker(self, handle: RayHandle, bin_pack: bool = True, detached: bool = False) -> RayHandle:

        handle._worker_args_ = vars(args()).copy()

        handle = handle._ray_initialize(
            self,
            bin_pack,
            detached,
            sharing_with = self._anchor_workers
        )
        handle.ready()

        self._worker_group_count += 1
        if self._anchor_workers is None:
            self._anchor_workers = handle._ray_workers
        
        return handle



class RayResourcePoolManager:
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.ngpus_per_node = getattr(args(), "ngpus_per_node", 8)
        self.nnodes = getattr(args(), "nnodes", 1)

        self.resources = [self.ngpus_per_node] * self.nnodes
        self.rest_resources = [self.ngpus_per_node] * self.nnodes

        self.allocated_gpus = 0
        self.all_gpus = sum(self.resources)

        self.pools: "list[RayResourcePool]" = []
    
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if (exc_type is None) and self.strict_mode:
            remaining_gpus = sum(self.rest_resources)
            if remaining_gpus > 0:
                raise RuntimeError(
                    f"Resource Budget Error: Not all GPUs were allocated! "
                    f"There are still {remaining_gpus} GPUs unallocated across nodes "
                    f"(Node status: {self.rest_resources}). "
                    f"All cluster GPUs must be fully claimed by resource pools."
                )
        return False

    def _check_resource_available(self):
        total_required_gpus = sum(
            sum(pool.nprocs) * pool.ngpus_per_proc
            for pool in self.pools
            if pool.pgs is None
        )

        if total_required_gpus == 0: return

        node_available_resources = (
            ray._private.state.available_resources_per_node()
        )

        total_available_gpus = sum(
            resources.get("GPU", 0)
            for resources in node_available_resources.values()
        )

        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than "
                f"pending required GPUs {total_required_gpus}"
            )

    def create_pool(self, nprocs: int, ncpus_per_proc: int = 1, ngpus_per_proc: int = 1 ,roles: list[RayRole] = []) -> RayResourcePool:
        # only support allocate ALL ngpus

        if type(nprocs) is not int or nprocs <= 0:
            raise ValueError("nprocs must be a positive integer.")
        if type(ngpus_per_proc) is not int or ngpus_per_proc <= 0:
            raise ValueError("ngpus_per_proc must be a positive integer.")

        if self.ngpus_per_node <= 0 or self.ngpus_per_node % ngpus_per_proc:
            raise ValueError(
                "ngpus_per_node must be positive and divisible by ngpus_per_proc."
            )
        workers_per_node = self.ngpus_per_node // ngpus_per_proc
        num_nodes, remainder = divmod(nprocs, workers_per_node)
        
        if remainder:
            raise ValueError(
                f"Whole-node allocation requires nprocs to be a multiple of "
                f"{workers_per_node}, got {nprocs}."
            )


        if num_nodes > self.rest_resources.count(self.ngpus_per_node):
            raise ValueError(
                f"Need {num_nodes} nodes, but remaining resources are "
                f"{self.rest_resources}."
            )
        nprocs = [workers_per_node] * num_nodes
        ngpus = [nproc * ngpus_per_proc for nproc in nprocs]

        for ngpu in ngpus:
            self.rest_resources.pop(self.rest_resources.index(ngpu))

        pool = RayResourcePool(
                pid = len(self.pools),
                nprocs = nprocs,
                ncpus_per_proc = ncpus_per_proc,
                ngpus_per_proc = ngpus_per_proc,
                roles = roles
            )
        self.pools.append(pool)
        self._check_resource_available()

        return pool
