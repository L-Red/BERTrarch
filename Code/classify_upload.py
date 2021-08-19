import sys

# import config
from dataloading import *
import pytorch_lightning as pl
from training.neural_nets import *
from training.data_preparation import *
import numpy as np
import pandas as pd

UCDP_LABELS = ['best']
ACLED_LABELS = []
WEIS_LABELS = []

def filetype(filename):
  return filename.rsplit('.', 1)[1].lower()

def anno_ucdp(filename):
  if filetype(filename) == 'csv':
    df = pd.read_csv(filename)
  elif filetype(filename) == 'xlsx':
    df = pd.read_excel(filename)
  else:
    return
  if len(df.columns) > 1:
    return
  tokenizer, bert_model = load_bert()
  tokens = tokenizer(df.tolist(), padding=True, return_tensors='pt', truncation=True)
  NUM_FEATURES = 300
  NUM_CLASSES = 100000
  INPUT_DIM = tokens.shape[1]
  return


def anno_acled(filname):
  return

def anno_weis(filname):
  return

def annotate(framework, filename):
  if framework == "UCDP":
    anno_ucdp(filename)
  elif framework == "ACLED":
    anno_acled(filename)
  else:
    anno_weis(filename)