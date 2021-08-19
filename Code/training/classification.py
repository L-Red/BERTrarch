class MulticlassClassification(nn.Module):

    def __init__(self, input_dim, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        
        self.embedding = nn.Embedding(input_dim, num_feature)

        self.layer_1 = nn.Linear(X.shape[1]*768, 1024).cuda()
        #self.layer_0 = nn.Linear(2048, 1024).cuda()
        self.layer_0_1 = nn.Linear(1024, 512).cuda()
        self.layer_1_2 = nn.Linear(512, 512).cuda()
        self.layer_2 = nn.Linear(512, 128).cuda()
        self.layer_3 = nn.Linear(128, 64).cuda()
        self.layer_out = nn.Linear(64, num_class).cuda()
        self.softmax = nn.Softmax(dim=1).cuda()

        self.relu = nn.ReLU().cuda()
        self.dropout = nn.Dropout(p=0.2).cuda()
        #self.batchnorm0 = nn.BatchNorm1d(2048).cuda()
        self.batchnorm0_1 = nn.BatchNorm1d(1024).cuda()
        self.batchnorm1 = nn.BatchNorm1d(512).cuda()
        self.batchnorm2 = nn.BatchNorm1d(128).cuda()
        self.batchnorm3 = nn.BatchNorm1d(64).cuda()

    def forward(self, x):
        #print(x.shape)
        #print(x.shape)
        x = bert_model(x)
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

def get_model(device, config):
  model = MulticlassClassification(input_dim = config["INPUT_DIM"], num_feature = config["NUM_FEATURES"], num_class=config["NUM_CLASSES"])
  model = model.to(device)
  #if(is_cuda):
  #    model.embedding.weight.data = pretrained_embeddings.cuda()
  #else:
  #    model.embedding.weight.data = pretrained_embeddings

  #criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
  optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
  return model, optimizer