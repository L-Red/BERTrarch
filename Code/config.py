import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import gridspec
import math
import sklearn
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
from torch.cuda.amp import GradScaler
global Dataset
from torch.utils.data import random_split, Dataset, WeightedRandomSampler, DataLoader
from sklearn.model_selection import train_test_split