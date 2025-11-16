import os
import sys
import yaml
os.environ["THEANO_FLAGS"] = "mode=FAST_RUN,optimizer=None,device=cpu,floatX=float32"
sys.path = [p for p in sys.path if ".local/lib/python3.8/site-packages" not in p]
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import itertools
import theano
import theano.tensor as T
import lasagne

from mypackage.models import FFN, CNN, CNN_LSTM, CNN_LSTM_Att
from mypackage.confusionmatrix import ConfusionMatrix
from mypackage.utils import iterate_minibatches, LSTMAttentionDecodeFeedbackLayer
from mypackage.metrics_mc import gorodkin, IC


if __name__ == '__main__':
    print()