"""
This is the classifier for conflict data. Choose to use ACLED, WEIS or UCDP
data to train the model
"""

import sys
import os

# import config
import numpy as np
from dataloading import *
import pytorch_lightning as pl
from training.neural_nets import *
from training.data_preparation import *

DEVICE = torch.device("cuda" if torch.cuda.is_available()else "cpu")
FRAMEWORK = ''
LABEL = ''
NUM_CLASSES=10000
# os.environ["TOKENIZERS_PARALLELISM"] = "true"

def classify(text, dtype):
    #WEIS
    if dtype == 'WEIS':
        return
    #UCDP
    if dtype == 'UCDP':
        return
    #ACLED
    if dtype == 'ACLED':
        return

def run_ucdp(label):   
    X, y, num_classes = load_translated_ucdp(label)
    return X, y, label, num_classes
    # UCDP = load_raw_ucdp()
    # UCDP2 = ucdp_process_raw(UCDP)
    # ucdp_save(UCDP2, '../UCDP/translated_df.csv')


def run_acled(label):
    X, y, num_classes = get_acled(label)
    return X, y, label, num_classes


def run_weis(label):
    X, y, num_classes = get_WEIS(label)
    return X, y, label, num_classes 

if sys.argv[1] == 'u': #run for UCDP
    #X, y = run_ucdp(sys.argv[2])
    X, y, LABEL, NUM_CLASSES = run_ucdp(sys.argv[2])
    FRAMEWORK = "ucdp"

elif sys.argv[1] == 'a': #run for ACLED
    X, y, LABEL, NUM_CLASSES = run_acled(sys.argv[2])
    FRAMEWORK = "acled"

elif sys.argv[1] == 'w': #run for WEIS
    X, y, LABEL, NUM_CLASSES = run_weis(sys.argv[2])
    FRAMEWORK = "weis"

else:
    try:
        print('no')
    except SystemError:
        print('run_classification.py <argument>')
        sys.exit(2)

# BERT
tokenizer, bert_model = load_bert(freeze=True)
y = np.array(y)
X, y = tokenize_bert(X, y, tokenizer)
split_list = split_data(X,y)
#normalize_data(split_list)
datasets = []
for tup in split_list:
    datasets.append(create_dataset(tup))
bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
config = dict(
    EPOCHS = 4,
    BATCH_SIZE = 32,
    LEARNING_RATE = 3e-2,
    NUM_FEATURES = 300,
    NUM_CLASSES = NUM_CLASSES,
    INPUT_DIM = X.shape[1],
    NUM_NODES = int(sys.argv[3])
    )
print(f"Input Dimensions: {X.shape[1]}")
data_module = DataModule(X, y, config['BATCH_SIZE'], datasets)
# weighted_sampler = make_weighted_sampler(datasets[0], distributed=False)

    #model = MulticlassClassification(config["INPUT_DIM"], config["NUM_FEATURES"], config["NUM_CLASSES"], "ucdp-bert-best-classified", bert_model)
n = torch.cuda.device_count()
model_name = f"{FRAMEWORK}-bert-{LABEL}-classified"
model = MulticlassClassification(config["INPUT_DIM"], config["NUM_FEATURES"], config["NUM_CLASSES"], model_name, bert_model, config)
train(model, config, data_module, DEVICE, reg=False)
