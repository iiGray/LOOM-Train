class AttrDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class LazyInitializeMeta(type):
    def __call__(cls, *args, **kwargs):
        obj = cls.__new__(cls)
        obj._stored_init_args_ = args
        obj._stored_init_kwargs_ = kwargs
        def perform_init(self):
            if not hasattr(self, "_stored_init_args_"): 
                return self
                raise RuntimeError(f"Re-initialization forbidden: {self} has already been intialized !!!")
            cls.__init__(self, *self._stored_init_args_, **self._stored_init_kwargs_)
            del self._stored_init_args_
            del self._stored_init_kwargs_

            return self

        obj._lazy_initialize_ = perform_init.__get__(obj)
        
        return obj