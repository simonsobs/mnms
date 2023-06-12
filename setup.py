from distutils.core import setup
import os

mnms_config_fn = os.path.join(os.environ['HOME'], '.mnms_config')
if not os.path.isfile(mnms_config_fn):
    with open(mnms_config_fn, 'a') as f:
        f.write("private_path: ''")

setup(name='mnms', packages=['mnms'], version='0.0.1')