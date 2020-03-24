import os

DATASET_NAME = "movie_lens"


COLUMN_DICT = None

MOVIE_LENS_COLUMN_DICT = {
    "item": "movieId",
    "user": "userId",
    "rating": "rating"
}

# to override in user_settings.py
PATH_TO_DATA = None
PATH_TO_PROJECT = None
# end of override

PROJECT_PATH = os.path.dirname(__file__)

try:
    from user_settings import *  # silence pyflakes
except ImportError:
    pass

if DATASET_NAME is "movie_lens":
    COLUMN_DICT = MOVIE_LENS_COLUMN_DICT

if COLUMN_DICT is not None:
    USER_COLUMN = COLUMN_DICT['user']
    ITEM_COLUMN = COLUMN_DICT['item']
    RATING_COLUMN = COLUMN_DICT['rating']

