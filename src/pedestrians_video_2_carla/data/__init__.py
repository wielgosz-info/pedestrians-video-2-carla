DATASETS_BASE = '/datasets'
OUTPUTS_BASE = '/outputs'  # for intermediate outputs, sine DATASETS should be read-only
DATA_MODULES = {}


def register_datamodule(name, datamodule_cls):
    """Register a data module class. This needs to be called in __init__.py of each data package."""
    DATA_MODULES[name] = datamodule_cls


def discover():
    """Discover registered data modules by walking the data directory."""

    import os
    from importlib import import_module
    from pkgutil import iter_modules

    for _, name, ispkg in iter_modules([os.path.dirname(__file__)], __name__ + '.'):
        if not ispkg:
            continue
        import_module(name)

    return DATA_MODULES
