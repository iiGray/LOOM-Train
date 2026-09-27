"""Load training dependencies only when a training API is requested.

The Ray controller can import ``linktrain.distributed`` without CUDA libraries.
Historical top-level exports (including ``from linktrain import *``) remain
available to training workers.
"""

from linktrain.core.modules import *
from importlib import import_module as _import_module

_loading_exports = False
def _load_exports():
    global _loading_exports, __all__
    if _loading_exports:
        return
    _loading_exports = True
    try:
        names = {'core', 'tasks'}
        for path in (
            'linktrain.core.utils.distributed',
            'linktrain.core.utils.common',
            'linktrain.core',
            'linktrain.tasks',
        ):
            module = _import_module(path)
            exports = getattr(module, '__all__', None)
            if exports is None:
                exports = [name for name in vars(module) if not name.startswith('_')]
            globals().update({name: getattr(module, name) for name in exports})
            names.update(exports)
        globals()['core'] = _import_module('linktrain.core')
        globals()['tasks'] = _import_module('linktrain.tasks')
        __all__ = sorted(names)
    finally:
        _loading_exports = False


def __getattr__(name):
    if name in {'core', 'tasks', 'distributed'}:
        module = _import_module(f'linktrain.{name}')
        globals()[name] = module
        return module
    if name == '__all__' or not name.startswith('_'):
        _load_exports()
        if name in globals():
            return globals()[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
