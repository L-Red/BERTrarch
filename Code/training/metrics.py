import torch
from torch import nn
def hit_at_k(pred, target, k):
    hits = []
    topk = torch.topk(pred, k=k)[1]
    for i, tens in enumerate(topk):
        if target[i] in tens:
            hits.append(1)
        else:
            hits.append(0)
    return sum(hits)/len(hits)


def MdAE(pred, target, reg=False):
    if not reg:
        pred = torch.argmax(pred, dim=1)
    return torch.median(torch.abs(target - pred))


def MdAPE(pred, target, reg=False):
    if not reg: 
        pred = torch.argmax(pred, dim=1)
    return torch.median(torch.abs((target - pred).float()/target.float()))


def num_loss(pred, target):
    cel = nn.CrossEntropyLoss(reduction='none')
    #target = torch.clamp(target.float(), min=1e-15, max=1-1e-15)
    #pred2 = torch.clamp(torch.argmax(pred, dim=1).double(), min=1e-11, max=None)
    pred2 = torch.add(torch.argmax(pred, dim=1), 1)
    target2 =  torch.add(target, 1)
    return torch.dot(((torch.log(target2.float()) - torch.log(pred2.float()))**2),cel(pred,target.long()))/pred.shape[0]


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)

    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)

    acc = torch.round(acc * 100)

    return acc