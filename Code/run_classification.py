"""
This is the classifier for conflict data. Choose to use ACLED, WEIS or UCDP
data to train the model
"""

import sys

# import config
from dataloading import *
import pytorch_lightning as pl
from training.neural_nets import *
from training.data_preparation import *
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available()else "cpu")
FRAMEWORK = ''
LABEL = ''


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
    X, y = load_translated_ucdp(label)
    return X, y, label
    # UCDP = load_raw_ucdp()
    # UCDP2 = ucdp_process_raw(UCDP)
    # ucdp_save(UCDP2, '../UCDP/translated_df.csv')


def run_acled():
    X, y = get_acled('fatalities')
    return X,y

if sys.argv[1] == 'u': #run for UCDP
    #X, y = run_ucdp(sys.argv[2])
    X, y, LABEL = run_ucdp(sys.argv[2])
    FRAMEWORK = "ucdp"

elif sys.argv[1] == 'a': #run for ACLED
    X, y = run_acled()
    FRAMEWORK = "acled"

elif sys.argv[1] == 'w': #run for WEIS
    X, y = run_weis()
    FRAMEWORK = "weis"

else:
    try:
        print('no')
    except SystemError:
        print('run_classification.py <argument>')
        sys.exit(2)

#BERT
tokenizer, bert_model = load_bert(freeze=False)
y = np.array(y)
X, y = tokenize_bert(X, y, tokenizer)
split_list = split_data(X,y)
#normalize_data(split_list)
datasets = []
for tup in split_list:
    datasets.append(create_dataset(tup))
config = dict(EPOCHS = 2,
              BATCH_SIZE = 32,
              LEARNING_RATE = 3e-2,
              NUM_FEATURES = 300,
              NUM_CLASSES = 100000,
              INPUT_DIM = X.shape[1])

del tokenizer, X, y
weighted_sampler = make_weighted_sampler(datasets[0], distributed=False)
data_module = DataModule(
                                            datasets[0],
                                            datasets[1],
                                            datasets[2],
                                            batchsize=config["BATCH_SIZE"],
                                            WEIGHTED_SAMPLER=True,
                                            weighted_sampler=weighted_sampler
                                            )

    #model = MulticlassClassification(config["INPUT_DIM"], config["NUM_FEATURES"], config["NUM_CLASSES"], "ucdp-bert-best-classified", bert_model)
n = torch.cuda.device_count()
model_name = f"{FRAMEWORK}-bert-{LABEL}-classified"
model = MulticlassClassification(config["INPUT_DIM"], config["NUM_FEATURES"], config["NUM_CLASSES"], model_name, bert_model, config)
train(model, config, data_module, DEVICE, reg=False)
save_model(model)
