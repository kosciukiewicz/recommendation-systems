import os

# to override in user_settings.py
PATH_TO_DATA = None
PATH_TO_PROJECT = None
# end of override

PROJECT_PATH = os.path.dirname(__file__)

try:
    from user_settings import *  # silence pyflakes
except ImportError:
    pass
