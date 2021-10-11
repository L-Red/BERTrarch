from sklearn.preprocessing import MinMaxScaler
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.data.distributed import DistributedSampler
from dataloading import *

def normalize_data(tuple_list):
  scaler = MinMaxScaler()
  tuple_list[0] = (scaler.fit_transform(tuple_list[0][0]), tuple_list[0][1])
  tuple_list[1] = (scaler.transform(tuple_list[1][0]), tuple_list[1][1])
  tuple_list[2] = (scaler.transform(tuple_list[2][0]), tuple_list[2][1])
  return

class DataModule(pl.LightningDataModule):
  def __init__(self, X, y, batch_size, datasets):
    super().__init__()
    self.X = X
    self.y = y
    self.datasets = datasets
    self.batchsize = batch_size
  # def prepare_data(self):
  #   #BERT
  #   tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', 'bert-base-uncased') 
  #   self.y = np.array(self.y)
  #   self.X, self.y = tokenize_bert(self.X, self.y, tokenizer)
  #   self.input_dim = self.X.shape[1]
  #   split_list = split_data(self.X,self.y)
  #   #normalize_data(split_list)
  #   for tup in split_list:
  #       self.datasets.append(create_dataset(tup))

  def setup(self, stage):
    self.train_dataloader = DataLoader(dataset=self.datasets[0],
                              batch_size=self.batchsize,
                              num_workers = 72
                              # sampler=self.weighted_sampler
    )
    self.val_dataloader = DataLoader(dataset=self.datasets[1], batch_size=self.batchsize, num_workers=72)
    self.test_dataloader = DataLoader(dataset=self.datasets[2], batch_size=self.batchsize, num_workers=72)

  def train_dataloader(self):
      return self.train_dataloader
  
  def val_dataloader(self):
      return self.val_dataloader
  
  def test_dataloader(self):
      return self.test_dataloader

def get_loaders(train_dataset, val_dataset, test_dataset, config, WEIGHTED_SAMPLER=False, weighted_sampler=None):
  if(WEIGHTED_SAMPLER):
    sampler = weighted_sampler
  else:
    sampler = None
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=config["BATCH_SIZE"],
                            sampler=sampler
  )
  val_loader = DataLoader(dataset=val_dataset, batch_size=config["BATCH_SIZE"])
  test_loader = DataLoader(dataset=test_dataset, batch_size=config["BATCH_SIZE"])
  
  data_module = DataModule(train_loader, val_loader, test_loader)
  return data_module

def make_weighted_sampler(train_dataset, distributed=False, rank=None, world_size=1):
  #Make weighted sampler, use target list for ACLED
  target_list = []
  class_counts = []
  indexer = {}
  ix_counter = 0
  for _, t in train_dataset:
      t = t.item()
      if t not in indexer:
          class_counts.append(0)
          indexer[t] = ix_counter
          ix_counter += 1
      class_counts[indexer[t]] += 1
      target_list.append(indexer[t])

  target_list = torch.tensor(target_list)

  # class_count_list = [0]*len(class_counts)
  # for i in range(len(class_counts)):
  #   class_count_list[i] = class_counts[indexer[i]]
  class_weights = (1./torch.tensor(list(class_counts), dtype=torch.float))

  class_weights_all = class_weights[target_list]

  weighted_sampler = WeightedRandomSampler(
      weights=class_weights_all,
      num_samples=len(class_weights_all),
      replacement=True
  )
  if distributed and torch.cuda.device_count() > 1:
    DistributedSampler(
        dataset, rank=rank, num_replicas=world_size, shuffle=True
    ) 
  return weighted_sampler