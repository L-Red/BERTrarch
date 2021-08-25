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



class BertRegression(nn.Module):
    def __init__(self, input_dim, num_feature, num_class, name, pretrained):
        super(BertRegression, self).__init__()
        self.pretrained = pretrained.cuda()
        self.name = name
        self.embedding = nn.Embedding(input_dim, num_feature)
        self.layer_1 = nn.Linear(input_dim*768, 512).cuda()
        self.layer_1_2 = nn.Linear(512, 512).cuda()
        self.layer_2 = nn.Linear(512, 128).cuda()
        self.layer_3 = nn.Linear(128, 64).cuda()
        self.layer_out = nn.Linear(64, 1).cuda()
        self.sig = nn.Sigmoid().cuda()
        self.dropout = nn.Dropout(p=0.2).cuda()
        self.batchnorm1 = nn.BatchNorm1d(512).cuda()
        self.batchnorm2 = nn.BatchNorm1d(128).cuda()
        self.batchnorm3 = nn.BatchNorm1d(64).cuda()

    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        with torch.no_grad():
            x = self.pretrained(x)
        x = x["last_hidden_state"]
        #print(x.shape)
        #x = x.view(BATCH_SIZE,-1)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)

        x = self.layer_1(x)
        #print(x.shape)
        #x = x.view(-1, 512,  45)
        #print(x.shape)
        x = self.batchnorm1(x)
        x = self.sig(x)
        
        x = self.layer_1_2(x)
        x = self.batchnorm1(x)
        x = self.sig(x)

        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.sig(x)
        x = self.dropout(x)

        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.sig(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)
        
        #in acled
        x = x.squeeze(1)
        return x

class MulticlassClassification(pl.LightningModule):
    def __init__(self, input_dim, num_feature, num_class, name, pretrained, config):
        super(MulticlassClassification, self).__init__()

        self.pretrained = pretrained

        self.name = name

        self.config = config

        self.criterion = nn.CrossEntropyLoss()
        
        self.embedding = nn.Embedding(input_dim, num_feature)

        self.layer_1 = nn.Linear(input_dim*768, 1024)
        #self.layer_0 = nn.Linear(2048, 1024)
        self.layer_0_1 = nn.Linear(1024, 512)
        self.layer_1_2 = nn.Linear(512, 512)
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
    
    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        x = self.pretrained(x)
        
        x = x["last_hidden_state"]
        #print(x.shape)
        #x = x.view(BATCH_SIZE,-1)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)

        x = self.layer_1(x)
        #print(x.shape)
        #x = x.view(-1, 512,  45)
        #print(x.shape)
        x = self.batchnorm0_1(x)
        x = self.relu(x)
        
        x = self.layer_0_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_1_2(x)
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
        
        mdae = MdAE(y_train_pred, y_train_batch, reg=False)
        mdape = MdAPE(y_train_pred, y_train_batch, reg=False)
        
        self.log("hit@1", hit_at_1)
        self.log("hit@3", hit_at_3)
        self.log("hit@10", hit_at_10)
        self.log("train_loss", train_loss)
        self.log("MdAE", mdae)
        self.log("MdAPE", mdape)

        #train_acc = multi_acc(y_train_pred, y_train_batch)
    def validation_step(self, val_batch, batch_idx):
        X_val_batch, y_val_batch = val_batch
        y_val_pred = self.forward(X_val_batch)
        val_loss = self.criterion(y_val_pred, y_val_batch)

        hit_at_1 = hit_at_k(y_val_pred, y_val_batch, 1)
        hit_at_3 = hit_at_k(y_val_pred, y_val_batch, 3)
        hit_at_10 = hit_at_k(y_val_pred, y_val_batch, 10)
        
        mdae = MdAE(y_val_pred, y_val_batch, reg=False)
        mdape = MdAPE(y_val_pred, y_val_batch, reg=False)

        #val_acc = multi_acc(y_val_pred, y_val_batch)
        
        #for prediction, truth in zip(y_val_pred, y_val_batch))
            #class_matrix_val[prediction][truth] += 1

        self.log("val_hit@3", hit_at_3)
        self.log("val_hit@1", hit_at_1)
        self.log("val_hit@10", hit_at_10)
        self.log("val_loss", val_loss)
        self.log("val_MdAE", mdae)
        self.log("val_MdAPE", mdape)

class MulticlassClassification1(nn.Module):
    def __init__(self, input_dim, num_feature, num_class, name, pretrained):
        super(MulticlassClassification1, self).__init__()

        self.pretrained = pretrained

        self.name = name

        self.train_dataloader = train_loader
        self.val_dataloader = val_loader
        self.test_dataloader = test_loader
        
        self.embedding = nn.Embedding(input_dim, num_feature)

        self.layer_1 = nn.Linear(input_dim*768, 1024)
        #self.layer_0 = nn.Linear(2048, 1024)
        self.layer_0_1 = nn.Linear(1024, 512)
        self.layer_1_2 = nn.Linear(512, 512)
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

    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        with torch.no_grad():
            x = self.pretrained(x)
        
        x = x["last_hidden_state"]
        #print(x.shape)
        #x = x.view(BATCH_SIZE,-1)
        x = torch.flatten(x, start_dim=1)
        #print(x.shape)

        x = self.layer_1(x)
        #print(x.shape)
        #x = x.view(-1, 512,  45)
        #print(x.shape)
        x = self.batchnorm0_1(x)
        x = self.relu(x)
        
        x = self.layer_0_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_1_2(x)
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
        #x = self.softmax(x)

        return x

        


def train(model, config, data_module, device, reg=False):
    wandb_logger = WandbLogger(project=model.name, config=config)
    n = torch.cuda.device_count()

    wandb_logger.watch(model)
    trainer = pl.Trainer(gpus=n, max_epochs=2, accelerator='dp', logger=wandb_logger, default_root_dir=f"./models/{model.name}")

    trainer.fit(model, datamodule=data_module)
    trainer.save_checkpoint(f"./checkpoints/{model.name}.ckpt")


def train1(model, config, data_module, device, reg=False):
    wandb.init(project=model.name, config=config)
    wandb.login()
    model = model.to(device)
    wandb.watch(model)
    counter = 0

    if reg:
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()
        
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    accuracy_stats = {
    'train': [],
    "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }

    scaler = GradScaler()

    from tqdm.notebook import tqdm
    print("Begin training.")
    for e in tqdm(range(1, config["EPOCHS"]+1)):

        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        for X_train_batch, y_train_batch in train_loader:
                #torch.cuda.empty_cache()
                X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():
                    y_train_pred = model(X_train_batch)
                    #print(f'y_train_pred: {y_train_pred.shape}')
                    
                    train_loss = criterion(y_train_pred, y_train_batch)
                #train_acc = multi_acc(y_train_pred, y_train_batch)
                
                #print(train_loss)
                
                #metrics
                if not reg:
                    hit_at_1 = hit_at_k(y_train_pred, y_train_batch, 1, reg)
                    hit_at_3 = hit_at_k(y_train_pred, y_train_batch, 3, reg)
                    hit_at_10 = hit_at_k(y_train_pred, y_train_batch, 10, reg)
                
                mdae = MdAE(y_train_pred, y_train_batch, reg)
                mdape = MdAPE(y_train_pred, y_train_batch, reg)
                
                if not reg: 
                    wandb.log({
                        "hit@1": hit_at_1, 
                        "hit@3": hit_at_3, 
                        "hit@10": hit_at_10,
                    })
                wandb.log({"train_loss": train_loss, 
                        "MdAE": mdae,
                        "MdAPE": mdape,
                        "epoch": e})
                
                #for prediction, truth in zip(y_train_pred.to('cpu'), y_train_batch.to('cpu')):
                    #class_matrix_train[prediction][truth] += 1

                scaler.scale(train_loss).backward()
                scaler.step(optimizer)

                train_epoch_loss += train_loss.detach().item()
                #train_epoch_acc += train_acc.item()
                
                
                scaler.update()

                if counter < 5:
                    print('skrr')
                    counter += 1


            # VALIDATION
        with torch.no_grad():

            val_epoch_loss = 0
            val_epoch_acc = 0

            model.eval()
            for X_val_batch, y_val_batch in val_loader:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)

                y_val_pred = model(X_val_batch)
                
                val_loss = criterion(y_val_pred, y_val_batch)

                if not reg:
                    hit_at_1 = hit_at_k(y_val_pred, y_val_batch, 1, reg)
                    hit_at_3 = hit_at_k(y_val_pred, y_val_batch, 3, reg)
                    hit_at_10 = hit_at_k(y_val_pred, y_val_batch, 10, reg)
                
                mdae = MdAE(y_val_pred, y_val_batch, reg)
                mdape = MdAPE(y_val_pred, y_val_batch, reg)

                #val_acc = multi_acc(y_val_pred, y_val_batch)
                
                #for prediction, truth in zip(y_val_pred, y_val_batch):
                    #class_matrix_val[prediction][truth] += 1

                val_epoch_loss += val_loss.item()
                #val_epoch_acc += val_acc.item()
                if not reg: 
                    wandb.log({
                        "hit@1": hit_at_1, 
                        "hit@3": hit_at_3, 
                        "hit@10": hit_at_10,
                    })
                wandb.log({"val_loss": val_loss, 
                        "epoch": e,
                        "MdAE": mdae,
                        "MdAPE": mdape,
                        })
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        #accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        #accuracy_stats['val'].append(val_epoch_acc/len(val_loader))


        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')

    

def save_model(model):
    torch.save(model.state_dict(), f'./models/{model.name}')
