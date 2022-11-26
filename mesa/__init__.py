
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

from numpy import array
from numpy.random import random
from pandas import DataFrame, read_hdf

import subprocess
from time import sleep
from os.path import isfile
import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logging.basicConfig(filename='runlog.log', encoding='utf-8', level=logging.DEBUG)
