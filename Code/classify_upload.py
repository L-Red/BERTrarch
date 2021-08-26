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
from sklearn.preprocessing import LabelEncoder 

UCDP_LABELS = ['best']
ACLED_LABELS = []
WEIS_LABELS = []

def anno_ucdp(filename):
  if filetype(filename) == 'csv':
    df = pd.read_csv(filename)[:10]
  elif filetype(filename) == 'xlsx':
    df = pd.read_excel(filename)
  else:
    return
  if len(df.columns) > 1:
    return
  tokenizer, bert_model = load_bert()
  tokens = tokenizer(df.iloc[:,0].tolist(), padding='max_length', max_length=512, return_tensors='pt', truncation=True)
  loader = DataLoader(tokens, batch_size=2)
  batch_size=2
  output_dataset = pd.DataFrame(columns=['best'])
  for label in UCDP_LABELS:
    encoder = LabelEncoder()
    encoder.classes_ = np.load(f'./label-encoder-classes/ucdp-{label}-classes.npy')
    NUM_FEATURES = 300
    NUM_CLASSES = len(encoder.classes_)
    tokens = tokens['input_ids']
    INPUT_DIM = tokens.shape[1]
    name = f"ucdp-bert-{label}-classified"
    MulticlassClassification(INPUT_DIM, NUM_CLASSES, NUM_CLASSES, name, bert_model, {})
    print(tokens.shape)
    model = MulticlassClassification.load_from_checkpoint(
        f"./checkpoints/{name}.ckpt", 
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
      output_data = encoder.inverse_transform(output_data.squeeze().numpy())
      output_dataset = pd.concat([output_dataset, pd.DataFrame({label: output_data})], ignore_index=True)
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