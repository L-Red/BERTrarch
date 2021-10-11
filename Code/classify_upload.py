import sys
import os.path

from flask import config

# import config
from dataloading import *
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from training.neural_nets import *
from training.data_preparation import *
from training.neural_nets import MulticlassClassification
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder 

UCDP_LABELS = ['best', 'side_a', 'side_b', 'event_clarity', 'type_of_violence']
ACLED_LABELS = []
WEIS_LABELS = []


def anno_ucdp(df):
  if len(df.columns) > 1:
    return
  tokenizer, bert_model = load_bert()
  print(df)
  tokens = tokenizer(df.iloc[:,0].tolist(), padding='max_length', max_length=512, return_tensors='pt', truncation=True)
  tokens = tokens['input_ids']
  batch_size=1
  loader = DataLoader(tokens, batch_size=batch_size)
  output_dataset = df
  for label in UCDP_LABELS:
    name = f"ucdp-bert-{label}-classified"
    if not os.path.isfile(f'../checkpoints/{name}.ckpt'):
      continue
    output_column = pd.DataFrame(columns=[label])
    encoder = LabelEncoder()
    encoder.classes_ = np.load(f'../label-encoder-classes/ucdp-{label}-classes.npy')
    NUM_FEATURES = 300
    NUM_CLASSES = len(encoder.classes_)
    INPUT_DIM = tokens.shape[1]
    config = {
      'BATCH_SIZE': batch_size
    }
    MulticlassClassification(INPUT_DIM, NUM_CLASSES, NUM_CLASSES, name, bert_model, config)
    print(tokens.shape)
    model = MulticlassClassification.load_from_checkpoint(
        f"../checkpoints/{name}.ckpt", 
        input_dim=INPUT_DIM, 
        num_feature=NUM_FEATURES,
        num_class=NUM_CLASSES,
        name=name,
        pretrained=bert_model,
        config={}
    )
    for i in range(0, tokens.shape[0], batch_size):
      input_data = tokens[i:i+batch_size]
      output_data = torch.argmax(model.forward(input_data), dim=1).view(1,-1)
      print(output_data.shape)
      if batch_size > 1:
        output_data = output_data.squeeze().numpy()
      else:
        output_data = output_data.numpy()
      output_data = encoder.inverse_transform(output_data)
      output_column = pd.concat([output_column, pd.DataFrame({label: output_data})], ignore_index=True)
    output_dataset = pd.concat([output_dataset, output_column], axis=1)
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