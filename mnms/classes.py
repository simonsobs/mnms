# These are the implemented noise model classes. They exist in the abstract
# in our minds, but the code is sliced into many levels of modules, classes
# etc, so we need to use a keyword that will track them across the repo.

### CONTRIBUTORS MAY ADD TO REGISTERED_CLASSES ###
REGISTERED_CLASSES = ['Tiled', 'Wavelet', 'FDW', 'Harmonic']

### DON'T MODIFY BELOW ###
def add_registry(baseclass):
    setattr(
        baseclass,
        '_subclass_registry',
        {c: None for c in REGISTERED_CLASSES}
        )
    setattr(baseclass, 'register_subclass', register_subclass)
    setattr(baseclass, 'get_subclass', get_subclass)
    return baseclass

@classmethod
def register_subclass(baseclass, subclass_key):
    try:
        assert baseclass._subclass_registry[subclass_key] is None, \
            f'{baseclass.__name__} already has registered subclass under {subclass_key}'
    except KeyError as e:
        raise ValueError(f'{subclass_key} not a valid mnms class, see classes.py') from e
    
    def decorator(subclass):
        baseclass._subclass_registry[subclass_key] = subclass
        setattr(subclass, '_noise_model_class', subclass_key)
        return subclass
    return decorator

@classmethod
def get_subclass(baseclass, subclass_key):
    return baseclass._subclass_registry[subclass_key]