import sys

# import config
from dataloading import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from training.neural_nets import *
from training.data_preparation import *
from training.neural_nets import MulticlassClassification
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
  loader = DataLoader(tokens, batch_size=1)
  output_dataset = pd.DataFrame()
  for label in UCDP_LABELS:
    NUM_FEATURES = 300
    NUM_CLASSES = 100000
    INPUT_DIM = tokens.shape[1]
    name = f"udcp-bert-{label}-classified"
    model = MulticlassClassification(INPUT_DIM, NUM_CLASSES, NUM_CLASSES, name, bert_model, {})
    model = model.load_from_checkpoint(f"./checkpoints/{name}.ckpt")
    for input_data in loader:
      output_data = model.forward(input_data)
      output_dataset[label].append(
        pd.Series(
          output_dataset,
          index=UCDP_LABELS
        ),
        ignore_index=True
      )
  return output_dataset


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