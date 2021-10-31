from flask import config
import wandb
from .metrics import *
import torch
import pytorch_lightning as pl
from torch.cuda.amp import GradScaler
from torch import nn
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.nn.parallel import DistributedDataParallel as DDP
from pytorch_lightning.loggers import WandbLogger
from torchmetrics.functional import f1

class MulticlassClassification(pl.LightningModule):
    def __init__(self, input_dim, num_feature, num_class, framework, label, freeze, pretrained, config):
        super(MulticlassClassification, self).__init__()

        self.pretrained = pretrained
        self.framework = framework
        self.label = label
        self.freeze = freeze
        if freeze:
            self.name = f"{framework}-bert_SVM_pooled_addedLayer-{label}-classified"
        else:
            self.name = f"{framework}-bert_pooled_addedLayer-{label}-classified"

        self.config = config

        self.num_classes = num_class

        self.criterion = nn.CrossEntropyLoss()
        
        self.embedding = nn.Embedding(input_dim, num_feature)

        self.layer_1 = nn.Linear(input_dim*768, 1024)
        self.hidden_layer = nn.Linear(768, 512)
        self.hidden_layer2 = nn.Linear(512, 256)
        self.classifier= nn.Linear(512, num_class)
        #self.layer_0 = nn.Linear(2048, 1024)
        self.layer_0_1 = nn.Linear(1024, 512)
        # self.layer_1_2 = nn.Linear(512, 512)
        self.layer_2 = nn.Linear(512, 128)
        self.layer_3 = nn.Linear(128, 64)
        self.layer_out = nn.Linear(64, num_class)
        self.softmax = nn.Softmax(dim=1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        #self.batchnorm0 = nn.BatchNorm1d(2048)
        self.batchnorm0_1 = nn.BatchNorm1d(1024)
        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(64)

        self.save_hyperparameters()
    
    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        
        #print(x.shape)
        #x = x.view(BATCH_SIZE,-1)
        #x = torch.flatten(x, start_dim=1)
        #print(x.shape)

        input_ids, attention_mask = x['input_ids'], x['attention_mask']      
        input_ids = input_ids.squeeze(dim=1)
        x = self.pretrained(input_ids, attention_mask=attention_mask)
        #changed from last_hidden_state to pooler_output
        #x = x["pooler_output"]
        x = torch.mean(x["last_hidden_state"],dim=1)

        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.classifier(x)
        #x = self.softmax(x)

        return x

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.config["LEARNING_RATE"])
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        X_train_batch, y_train_batch = train_batch
        y_train_pred = self.forward(X_train_batch)
        #print(f'y_train_pred: {y_train_pred.shape}')
        train_loss = self.criterion(y_train_pred, y_train_batch)
        
        hit_at_1 = hit_at_k(y_train_pred, y_train_batch, 1)
        hit_at_3 = hit_at_k(y_train_pred, y_train_batch, 3)
        hit_at_10 = hit_at_k(y_train_pred, y_train_batch, 10)
        f1score = f1(y_train_pred, y_train_batch, num_classes=self.num_classes, average='weighted')
        
        mdae = MdAE(y_train_pred, y_train_batch, reg=False)
        mdape = MdAPE(y_train_pred, y_train_batch, reg=False)
        
        self.log("hit@1", hit_at_1)
        self.log("hit@3", hit_at_3)
        self.log("hit@10", hit_at_10)
        self.log("train_loss", train_loss)
        self.log("MdAE", mdae)
        self.log("MdAPE", mdape)
        self.log("F1", f1score)
        return train_loss

        #train_acc = multi_acc(y_train_pred, y_train_batch)
    def validation_step(self, val_batch, batch_idx):
        X_val_batch, y_val_batch = val_batch
        y_val_pred = self.forward(X_val_batch)
        val_loss = self.criterion(y_val_pred, y_val_batch)

        hit_at_1 = hit_at_k(y_val_pred, y_val_batch, 1)
        hit_at_3 = hit_at_k(y_val_pred, y_val_batch, 3)
        hit_at_10 = hit_at_k(y_val_pred, y_val_batch, 10)

        f1score = f1(y_val_pred, y_val_batch, num_classes=self.num_classes, average='weighted')
        
        mdae = MdAE(y_val_pred, y_val_batch, reg=False)
        mdape = MdAPE(y_val_pred, y_val_batch, reg=False)

        self.log("val_hit@3", hit_at_3)
        self.log("val_hit@1", hit_at_1)
        self.log("val_hit@10", hit_at_10)
        self.log("val_loss", val_loss)
        self.log("val_MdAE", mdae)
        self.log("val_MdAPE", mdape)
        self.log("val_F1", f1score)
        
        return val_loss

    def test_step(self, test_batch, batch_idx):
        X_test_batch, y_test_batch = test_batch
        y_test_pred = self.forward(X_test_batch)
        test_loss = self.criterion(y_test_pred, y_test_batch)

        hit_at_1 = hit_at_k(y_test_pred, y_test_batch, 1)
        hit_at_3 = hit_at_k(y_test_pred, y_test_batch, 3)
        hit_at_10 = hit_at_k(y_test_pred, y_test_batch, 10)
        
        mdae = MdAE(y_test_pred, y_test_batch, reg=False)
        mdape = MdAPE(y_test_pred, y_test_batch, reg=False)

        f1score = f1(y_test_pred, y_test_batch, num_classes=self.num_classes, average='weighted')

        self.log("test_loss", test_loss)
        self.log("test_F1", f1score)

        return {"pred": y_test_pred, "truth": y_test_batch}

    def test_step_end(self, batch_parts):
        predictions = batch_parts["pred"]
        truths = batch_parts["truth"]
        return (predictions, truths)
    

    def test_epoch_end(self, test_step_outputs):
        predictions = []
        truths = []
        for out in test_step_outputs:
            predictions.append(out[0])
            truths.append(out[1])
        if predictions[0].shape[0] != predictions[len(predictions)-1].shape[0]:
            predictions = predictions[0:len(predictions)-1]
            truths = truths[0:len(truths)-1]
        predictions = torch.cat(predictions)
        truths = torch.cat(truths)
        f1score = f1(predictions, truths, num_classes=self.num_classes, average='weighted')
        print(f1score)

        #val_acc = multi_acc(y_val_pred, y_val_batch)
        
        if self.freeze:
            f = open(f"/cluster/work/cotterell/liaroth/bachelor-thesis/Results/{self.framework}_{self.label}.txt", "a")
            f.write(f"BERT SVM:\n{f1score}\n")
            f.close()
        else:
            f = open(f"/cluster/work/cotterell/liaroth/bachelor-thesis/Results/{self.framework}_{self.label}.txt", "a")
            f.write(f"BERT fine-tuned:\n{f1score}\n")
            f.close()

        #for prediction, truth in zip(y_val_pred, y_val_batch))
            #class_matrix_val[prediction][truth] += 1
        return f1score       


def train(model, config, data_module, device, reg=False):
    wandb_logger = WandbLogger(project=model.name, config=config)
    n = torch.cuda.device_count()

    wandb_logger.watch(model)
    if config['NUM_NODES'] > 1:
        parallellization = 'ddp'
    else:
        parallellization = 'dp'
    trainer = pl.Trainer(
        gpus=-1, 
        max_epochs=config['EPOCHS'], 
        accelerator=parallellization, 
        num_nodes=config['NUM_NODES'], 
        logger=wandb_logger, 
        default_root_dir=f"./models/{model.name}",
        # accumulate_grad_batches=10,
        # gradient_clip_val=0.5,
        # stochastic_weight_avg=True
        )

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(f"./checkpoints/{model.name}.ckpt")
    trainer.test(ckpt_path="best")

  

def save_model(model):
    torch.save(model.state_dict(), f'./models/{model.name}')
