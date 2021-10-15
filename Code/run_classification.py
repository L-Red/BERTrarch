"""
This is the classifier for conflict data. Choose to use ACLED, WEIS or UCDP
data to train the model
"""

import sys
import os

# import config
import numpy as np
from dataloading import *
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from training.neural_nets import *
from training.data_preparation import *
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
import torchtext
from torchtext.data import get_tokenizer
from torchtext.vocab import GloVe
from torchmetrics.functional import f1
DEVICE = torch.device("cuda" if torch.cuda.is_available()else "cpu")
FRAMEWORK = ''
LABEL = ''
NUM_CLASSES=10000
# os.environ["TOKENIZERS_PARALLELISM"] = "true"
EPOCHS = 4
BATCH_SIZE = 64

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

#https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        try:
            yield torch.tensor(lst[i:i + n])
        except IndexError:
            yield torch.tensor(lst[i:len(lst)])

def train_SVM(X, y):
    split_list = split_data(X,y)
    X_train, y_train = split_list[0]
    X_val, y_val = split_list[1]
    X_test, y_test = split_list[2]
    clf = svm.LinearSVC()
    #train
    clf.fit(X_train,y_train)
    #validate
    val_pred = clf.predict(X_val)
    score = f1_score(y_val, val_pred, average='weighted')
    print(f'Validation F1_Score : {score}')
    #test
    test_pred = clf.predict(X_test)
    score = f1_score(y_test, test_pred, average='weighted')
    print(f'Test F1_Score: {score}')

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

#Majority Vote classifier
if sys.argv[4] == 'maj':
    #find majority class within labels
    values, counts = np.unique(y, return_counts=True)
    ind = np.argmax(counts)
    majority_class = values[ind]
    _,y_test = split_data(X,y)[2]
    #predict everythin with majority class and do f1
    y_pred = np.array([majority_class]*len(y_test))
    score = f1_score(y_test, y_pred, average='weighted')
    print(f'Majority F1 score {score}')

#Bag Of Words classifier
elif sys.argv[4] == 'svm':
    vectorizer = CountVectorizer(ngram_range=(1,5))
    X = vectorizer.fit_transform(X)  
    train_SVM(X,y)

#GloVe classifier
elif sys.argv[4] == 'glove':
    create_embedding = False
    if create_embedding:
        embedding = GloVe()
        tokenizer = get_tokenizer("basic_english")
        gloveX = []
        for x in X:
            tokenizedX = tokenizer(x)
            gloveX.append(embedding.get_vecs_by_tokens(tokenizedX).tolist())
        print(f'size of gloveX: {len(gloveX)}')
        print(gloveX[0])
        df = pd.DataFrame(gloveX)
        df.to_csv(f'../GloVe/{FRAMEWORK}.csv')
    else:
        del X
        df = pd.read_csv(f'../GloVe/{FRAMEWORK}.csv', index_col=0)

        #https://stackoverflow.com/a/45425158
        import json
        df = df.apply(lambda x: x.dropna().apply(json.loads))

        np_array = df.apply(lambda x: x.apply(np.asarray)).to_numpy()

        vfunc = np.vectorize(isinstance)
        X = np.zeros(
            (
                np_array.shape[0],
                #np_array.shape[1], 
                np_array[0,0].size
                )
            )
        for i in range(np_array.shape[0]):
            ma = np.ma.MaskedArray(np_array[i], mask=vfunc(np_array[i],float))
            av = np.average(ma).data
            """mask = vfunc(np_array[i],float)
            X[i][mask] = av
            for j in range(X[i].shape[0]):
                if not mask[j]:
                    X[i][j] = np_array[i][j]
            """
            X[i] = av
        #X = X.reshape(X.shape[0], -1)

        print(f'First entry in X: {X[0]}')
        train_SVM(X,y)






else:
    if sys.argv[4] == 'bert-svm':
        #non fine-tuned BERT
        tokenizer, bert_model = load_bert(freeze=True)
    else:
        #fine-tuned BERT
        tokenizer, bert_model = load_bert(freeze=False)
    #y = np.array(y)
    #X = np.array(X)
    #X, y = tokenize_bert(X, y, tokenizer)
    max_length = 64
    split_list = split_data(X,y)
    #normalize_data(split_list)
    datasets = []
    for tup in split_list:
        datasets.append(create_dataset(tup, max_length))
    # bert_model = torch.hub.load('huggingface/pytorch-transformers', 'model', 'bert-base-uncased')
    config = dict(
        EPOCHS = EPOCHS,
        BATCH_SIZE = BATCH_SIZE,
        LEARNING_RATE = 3e-2,
        NUM_FEATURES = 300,
        NUM_CLASSES = NUM_CLASSES,
        INPUT_DIM = max_length,
        NUM_NODES = int(sys.argv[3])
        )
    data_module = DataModule(X, y, config['BATCH_SIZE'], datasets)
    # weighted_sampler = make_weighted_sampler(datasets[0], distributed=False)

        #model = MulticlassClassification(config["INPUT_DIM"], config["NUM_FEATURES"], config["NUM_CLASSES"], "ucdp-bert-best-classified", bert_model)
    n = torch.cuda.device_count()
    model_name = f"{FRAMEWORK}-bert_pooled_addedLayer-{LABEL}-classified"
    model = MulticlassClassification(config["INPUT_DIM"], config["NUM_FEATURES"], config["NUM_CLASSES"], model_name, bert_model, config)
    train(model, config, data_module, DEVICE, reg=False)
