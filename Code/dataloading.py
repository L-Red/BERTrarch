from numpy.core.defchararray import encode
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import pandas as pd
from langdetect import detect
from easynmt import EasyNMT
from sklearn.preprocessing import LabelEncoder
import numpy as np

#load ACLED data
def get_acled(label):
  ACLED = pd.read_csv('../ACLED/ACLED_all_25_05_2021.csv', engine='python', skiprows=range(1,1000), nrows=250000, header=0)
  y = ACLED[label].tolist()
  le = LabelEncoder()
  y = le.fit_transform(y)
  np.save(f'./label-encoder-classes/acled-{label}-classes.npy', le.classes_)
  text_list = ACLED['notes'].tolist()
  num_classes = len(le.classes_)
  amt = len(y)
  return (text_list, y, num_classes)


#load UCDP data
def load_raw_ucdp():
  return pd.read_csv('../UCDP/UCDP_data.csv')

def load_translated_ucdp(label):
  # #load UCDP DATA
  # with open("../UCDP/translated_ucdp_data.txt", "rb") as fp:   # Unpickling
  #   text_list = pickle.load(fp)
      
  # with open("../UCDP/translated_ucdp_best-labels.txt", "rb") as fp:   # Unpickling
  #   y_list = pickle.load(fp)

  text_df = pd.read_csv('../UCDP/translated_df.csv')
  text_df = text_df[['source_headline', label]].dropna()
  
  text_list = text_df['source_headline'].tolist()
  y_list = text_df[label].tolist()

  le = LabelEncoder()
  y_list = le.fit_transform(y_list)
  np.save(f'./label-encoder-classes/ucdp-{label}-classes.npy', le.classes_)
  num_classes = len(le.classes_)

  #delete all non-string entries
  ls = []
  for index, li in enumerate(text_list):
      if not isinstance(li, str):
          ls.append(index)
  ls.sort()
  ls.reverse()
  for index in ls:
      text_list.pop(index)
      y_list.pop(index)

  return (text_list, y_list, num_classes)

def ucdp_process_raw(UCDP):
  from langdetect import detect
  from easynmt import EasyNMT

  trans_model = EasyNMT('m2m_100_418M')

  text_df = UCDP
  #drop all titles where they are INCIDENTS or TS
  text_df.drop(text_df[(text_df.source_headline.str.startswith('TS')) | (text_df.source_headline.str.startswith('INCIDENT'))].index, inplace=True)
  #tanslate non-english titles
  text_df['source_headline'].apply(trans_non_en)
  

  return text_df

def ucdp_save(text_df, PATH):
  text_df.to_csv(PATH)
  
  
def trans_non_en(s):
    if isinstance(s, str):
        try:
            if not (detect(s) == 'en'):
                #print(s)
                new_s = trans_model.translate(s, target_lang = "en")
                #print(new_s)
                return new_s
        except:
            print('not translatable')
            return s


#load WEIS data
def get_WEIS(label):
    WEIS = pd.read_csv('../WEIS/weis_data_decoded.csv')
    # nltk.download('stopwords')
    y = WEIS[label].tolist()
    le = LabelEncoder()
    y_list = le.fit_transform(y)
    np.save(f'./label-encoder-classes/weis-{label}-classes.npy', le.classes_)
    text_list = WEIS['text'].tolist()
    num_classes = len(le.classes_)
    return (text_list, y_list, num_classes)


#process the data
def load_bert(freeze=True):
  tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased')    # Download vocabulary from S3 and cache.
  bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
  if(freeze):
    for param in bert_model.parameters():
      param.requires_grad = False
  return (tokenizer, bert_model)   # Download model and configuration from S3 and cache

def tokenize_bert(text_list, y_list, tokenizer, cut=1):
  indexed_tokens = tokenizer(
    text_list[0:len(text_list)//cut], 
    padding=True, 
    return_tensors='np', 
    truncation=True, 
    max_length=64
    )
  y = y_list[0:len(text_list)//cut]
  X = indexed_tokens['input_ids']
  return (X, y)

def split_data(X, y):
  from torch.utils.data import random_split, Dataset, WeightedRandomSampler
  from sklearn.model_selection import train_test_split
  # l = X.shape[0]
  # X_train, X_test, X_val = random_split(X, [l*7//10, l*2//10, l-l*9//10], generator=torch.Generator().manual_seed(42))
  # y_train, y_test, y_val = random_split(y, [l*7//10, l*2//10, l-l*9//10], generator=torch.Generator().manual_seed(42))

  # Split into train+val and test
  X_train, X_testval, y_train, y_testval = train_test_split(X, y, test_size=0.2, random_state=69)

  # Split train into train-val
  X_test, X_val, y_test, y_val = train_test_split(X_testval, y_testval, test_size=0.5, random_state=21) #previously trainval, 0.1 here
  return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]
  

def create_dataset(tup):
  return ClassifierDataset(*tup)

  print(X_train.shape)
  return (train_dataset, val_dataset, test_dataset)
  #Define dataset for dataloader
class ClassifierDataset(Dataset):

  def __init__(self, X_data, y_data):
      self.X_data = torch.from_numpy(X_data)
      self.y_data = torch.from_numpy(y_data)

  def __getitem__(self, index):
      return self.X_data[index], self.y_data[index]

  def __len__ (self):
      return len(self.X_data)


def eligible_dt(y_list):
  return not (isinstance(y_list[0], numpy.float64) or isinstance(y_list[0], numpy.float32) or isinstance(y_list[0], numpy.float16) or isinstance(y_list[0], numpy.complex64) or isinstance(y_list[0], numpy.complex128) or isinstance(y_list[0], numpy.int64) or isinstance(y_list[0], numpy.int32) or isinstance(y_list[0], numpy.int16) or isinstance(y_list[0], numpy.int8) or isinstance(y_list[0], numpy.uint8) or isinstance(y_list[0], bool))