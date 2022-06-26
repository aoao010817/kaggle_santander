import math
import io
import gzip
import pickle
import zlib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import engines
from utils import *

np.random.seed(2022)
transformers = {}

fname = "../input/8th.clean.all.csv"
train_df = pd.read_csv(fname, dtype=dtypes)