from setuptools import setup
import os

# add an mnms config file with empty private path
mnms_config_fn = os.path.join(os.environ['HOME'], '.mnms_config.yaml')
if not os.path.isfile(mnms_config_fn):
    with open(mnms_config_fn, 'a') as f:
        f.write("private_path: ''\n")

setup(
    name='mnms',
    packages=['mnms'],
    version='0.0.10',
    install_requires=[
        'ducc0>=0.30.0',
        'numba',
        'pixell>=0.21',
        'sofind>=0.0.11'
    ]
    )
