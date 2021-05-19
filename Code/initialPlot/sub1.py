import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import gridspec
import math

from gensim.corpora import Dictionary
from gensim.models.ldamodel import LdaModel
from gensim.models import CoherenceModel, AuthorTopicModel
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from datetime import datetime
from torch.autograd import Variable
from collections import Counter
from torchtext.vocab import Vocab

WEIS = pd.read_csv('../../WEIS/weis_data_decoded.csv')
nltk.download('stopwords')

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def prepare_text(data=WEIS, column='text'):
    text_raw = data[column].copy()
    en_stop_words = set(stopwords.words('english'))

    # remove stopwords
    text_prep = text_raw.apply(lambda row: ' '.join([word for word in row.split() if (not word in en_stop_words)]))

    # tokenize each newspaper title
    tokenizer = RegexpTokenizer('[a-zA-Z]\w+\'?\w*')
    text_prep = text_prep.apply(lambda row: tokenizer.tokenize(row))

    return text_prep

tokenized = prepare_text()

is_cuda = torch.cuda.is_available()
print("Cuda Status on system is {}".format(is_cuda))

from torchtext.vocab import GloVe
embedding_glove = GloVe()

gold_class = {}
gold_count = 0
for i in WEIS['goldstein']:
    if i not in gold_class:
        gold_class[i] = gold_count
        gold_count += 1
nr_classes = len(gold_class)

def gold_to_class(gold):
    return gold_class[gold]
WEIS['goldclass'] = WEIS['goldstein'].apply(gold_to_class)

#plot distribution
class_counts = [0]*nr_classes
for ix in WEIS['goldclass']:
    class_counts[ix] += 1
plt.bar(range(nr_classes), class_counts)
plt.title('Goldstein Class Frequency')
plt.savefig('../goldstein_class_frequency.png')

#from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pad_sequence
y = WEIS['goldclass'].to_numpy()
#Xx = pad_sequence([embedding_glove.get_vecs_by_tokens(x) for x in tokenized], batch_first=True)
def lookup(y):
    try:
        return embedding_glove.stoi[y]
    except:
        return 0
tokenized2 = [torch.tensor([lookup(y) for y in x]) for x in tokenized]
X = pad_sequence(tokenized2, batch_first=True)

from torch.utils.data import random_split, Dataset, WeightedRandomSampler, DataLoader
from sklearn.model_selection import train_test_split
# l = X.shape[0]
# X_train, X_test, X_val = random_split(X, [l*7//10, l*2//10, l-l*9//10], generator=torch.Generator().manual_seed(42))
# y_train, y_test, y_val = random_split(y, [l*7//10, l*2//10, l-l*9//10], generator=torch.Generator().manual_seed(42))

# Split into train+val and test
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=69)

# Split train into train-val
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)

print(X_train.shape)
#Define dataset for dataloader
class ClassifierDataset(Dataset):

    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__ (self):
        return len(self.X_data)


train_dataset = ClassifierDataset(X_train, y_train)
val_dataset = ClassifierDataset(X_val, y_val)
test_dataset = ClassifierDataset(X_test, y_test)

#Make weighted sampler, use target list
target_list = []
for _, t in train_dataset:
    target_list.append(t)

target_list = torch.tensor(target_list)
target_list = target_list[torch.randperm(len(target_list))]


#make class weight
class_weights = 1./torch.tensor(class_counts, dtype=torch.float).to(device)

class_weights_all = class_weights[target_list]
weighted_sampler = WeightedRandomSampler(
    weights=class_weights_all,
    num_samples=len(class_weights_all),
    replacement=True
)

#SETTING PARAMETERS
EPOCHS = 3
BATCH_SIZE = 2
LEARNING_RATE = 0.0007
NUM_FEATURES = 300
NUM_CLASSES = len(gold_class)
INPUT_DIM = X_train.shape[1]

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          sampler=weighted_sampler
)
val_loader = DataLoader(dataset=val_dataset, batch_size=1)
test_loader = DataLoader(dataset=test_dataset, batch_size=1)


class MulticlassClassification(nn.Module):
    def __init__(self, input_dim, num_feature, num_class):
        super(MulticlassClassification, self).__init__()

        self.embedding = nn.Embedding(input_dim, num_feature)

        self.layer_1 = nn.Linear(num_feature * 45, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

    def forward(self, x):
        # print(x.shape)
        print(x.shape)
        x = self.embedding(x)
        print(x.shape)
        x = x.view(BATCH_SIZE, -1)
        print(x.shape)

        x = self.layer_1(x)
        print(x.shape)
        # x = x.view(-1, 512,  45)
        # print(x.shape)
        x = self.batchnorm1(x)
        x = self.relu(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.layer_out(x)

        return x

pretrained_embeddings = embedding_glove.vectors

print(pretrained_embeddings.shape)


model = MulticlassClassification(input_dim = INPUT_DIM, num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
model.to(device)

model.embedding.weight.data = pretrained_embeddings.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
print(model)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc



accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


from tqdm.notebook import tqdm
print("Begin training.")
for e in tqdm(range(1, EPOCHS+1)):

    # TRAINING
    train_epoch_loss = 0
    train_epoch_acc = 0
    model.train()
    for X_train_batch, y_train_batch in train_loader:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()

            y_train_pred = model(X_train_batch)
            #print(f'y_train_pred: {y_train_pred.shape}')

            train_loss = criterion(y_train_pred, y_train_batch.long())
            train_acc = multi_acc(y_train_pred, y_train_batch)

            train_loss.backward()
            optimizer.step()

            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()


        # VALIDATION
    with torch.no_grad():

        val_epoch_loss = 0
        val_epoch_acc = 0

        model.eval()
        for X_val_batch, y_val_batch in val_loader:
            X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

            y_val_pred = model(X_val_batch)

            val_loss = criterion(y_val_pred, y_val_batch)
            val_acc = multi_acc(y_val_pred, y_val_batch)

            val_epoch_loss += val_loss.item()
            val_epoch_acc += val_acc.item()
    loss_stats['train'].append(train_epoch_loss/len(train_loader))
    loss_stats['val'].append(val_epoch_loss/len(val_loader))
    accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
    accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


    print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

torch.save(model, './model_glove.pt')
