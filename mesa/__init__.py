import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import logging

logging.basicConfig(encoding='utf-8', level=logging.INFO)
logging.basicConfig(filename='runlog.log', encoding='utf-8', level=logging.DEBUG)
