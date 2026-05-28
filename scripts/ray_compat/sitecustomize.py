# Ray 2.3.0 + Python 3.11 + numpy 2.0 compatibility shims
# Loaded in every Python process via PYTHONPATH=/tmp/ray_compat
import sys
import types

# 1. numpy 2.0 removed numeric type aliases
import numpy as np
for _a, _v in [
    ('bool8',   np.bool_),
    ('float',   float),
    ('int',     int),
    ('complex', complex),
    ('product', np.prod),   # removed in numpy 2.0
    ('cumproduct', np.cumprod),
    ('sometrue',  np.any),
    ('alltrue',   np.all),
]:
    if not hasattr(np, _a): setattr(np, _a, _v)


# 2. Stub out ray.data / ray.train to prevent Python 3.11 dataclass crash.
#    The stub must be picklable so Ray cloudpickle can handle it across processes.

def _load_stub_module(name):
    """Factory used by __reduce__ – returns existing or creates new stub."""
    if name in sys.modules:
        return sys.modules[name]
    return _make_stub(name)


def _make_stub(fullname):
    m = _StubModule(fullname)
    m.__file__ = '<stub>'
    m.__path__ = []
    m.__package__ = fullname.rpartition('.')[0] or fullname
    m.__spec__ = None
    sys.modules[fullname] = m
    return m


class _StubModule(types.ModuleType):
    """Stub module that lazily creates child stubs on attribute access."""

    def __getattr__(self, name):
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        fullname = self.__name__ + '.' + name
        child = _make_stub(fullname)
        object.__setattr__(self, name, child)
        return child

    def __reduce__(self):
        # Tell pickle how to reconstruct this object in another process.
        # _load_stub_module is a module-level function → picklable.
        return (_load_stub_module, (self.__name__,))


class _RayDataStubFinder:
    """sys.meta_path hook – intercepts ray.data.* / ray.train.* imports."""
    _PREFIXES = ('ray.data', 'ray.train')

    def find_module(self, fullname, path=None):
        if any(fullname == p or fullname.startswith(p + '.') for p in self._PREFIXES):
            return self
        return None

    def load_module(self, fullname):
        if fullname in sys.modules:
            return sys.modules[fullname]
        return _make_stub(fullname)

    def __reduce__(self):
        return (_RayDataStubFinder, ())


sys.meta_path.insert(0, _RayDataStubFinder())
