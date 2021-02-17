import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.distance import distance
import torch.nn.functional as F


def pairwise_loss(output, label, alpha=10.0, class_num=5.0, l_threshold=15.0):
    '''https://github.com/thuml/HashNet/issues/27#issuecomment-494265209'''
    bits = output.shape[1]
    similarity = Variable(torch.mm(label.data.float(), label.data.float().t()) > 0).float()
    dot_product = alpha * torch.mm(output, output.t()) / bits
    mask_dot = dot_product.data > l_threshold
    mask_exp = dot_product.data <= l_threshold
    mask_positive = similarity.data > 0
    mask_negative = similarity.data <= 0
    mask_dp = mask_dot & mask_positive
    mask_dn = mask_dot & mask_negative
    mask_ep = mask_exp & mask_positive
    mask_en = mask_exp & mask_negative

    dot_loss = dot_product * (1-similarity)
    exp_loss = torch.log(1+torch.exp(dot_product)) - similarity * dot_product
    loss = (torch.sum(torch.masked_select(exp_loss, Variable(mask_ep))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dp)))) * class_num + \
            torch.sum(torch.masked_select(exp_loss, Variable(mask_en))) + torch.sum(torch.masked_select(dot_loss, Variable(mask_dn)))

    return loss / (torch.sum(mask_positive.float()) * class_num + torch.sum(mask_negative.float()))


def pairwise_loss_2(output, label, alpha=5.0):
    batch_size, bits = output.shape
    mask = (torch.eye(batch_size) == 0).to(torch.device("cuda"))
    S =  torch.mm(label.float(), label.float().t())
    S_m = torch.masked_select(S, mask)
    ip = alpha * torch.mm(output, output.t()) / bits
    ip_m = torch.masked_select(ip, mask)
    loss_1  = - (S_m * ip_m - torch.log(1 + torch.exp(ip_m)))
    loss = loss_1.mean()
    return loss


def contrastive_loss(output, label, margin=2):
    '''contrastive loss
    - Deep Supervised Hashing for Fast Image Retrieval
    '''
    batch_size = output.shape[0]
    S =  torch.mm(label.float(), label.float().t())
    dist = distance(output, dist_type='euclidean2')
    loss_1 = S * dist + (1 - S) * torch.max(margin - dist, torch.zeros_like(dist))
    loss = torch.sum(loss_1) / (batch_size*(batch_size-1))
    return loss


# weights = torch.tensor([0.50857143, 0.61952381, 0.89038095, 0.70780952, 0.89171429,
#                         0.85942857, 0.89714286, 0.9067619 , 0.8847619 , 0.85714286,
#                         0.87914286, 0.9187619 , 0.92685714, 0.90457143, 0.904     ,
#                         0.91561905, 0.92561905, 0.9272381 , 0.92457143, 0.91742857,
#                         0.90780952]).cuda()

def hadamard_loss(output, label, hadamard):
    def rand_num():
        r = torch.round(torch.rand(1))
        return r if r == 1 else torch.tensor([-1])

    batch_size, bit = output.shape
    label = label.float()
    if torch.sum(label) == batch_size:
        hc = torch.mm(label, hadamard)
    else:
        # label *= weights
        hc = torch.mm(label, hadamard)
        # hc[hc>0] = 1
        # hc[hc<0] = -1
        # hc[hc==0] = rand_num()
    loss = (output - hc) ** 2
    return loss.mean()


def quantization_loss(output, square=True):
    if square:
        loss = torch.mean((torch.abs(output) - 1) ** 2)
    else:
        loss = torch.mean(torch.abs((torch.abs(output) - 1)))
    return loss


def balance_loss(output):
    '''balance loss

    Each bit should be half 0 and half 1.
    - Supervised Learning of Semantics-preserving Hashing via Deep Neural Networks for Large-scale Image Search
    '''
    H = torch.sign(output)
    H_mean = torch.mean(H, dim=0)
    loss = torch.mean(H_mean ** 2)
    return loss


def independence_loss(output):
    '''independence loss
    - Deep Triplet Quantization
    '''
    batch_size, bit = output.shape
    H = torch.sign(output)
    I = torch.eye(bit).cuda()
    loss = torch.mean((torch.mm(H.t(), H) / batch_size - I) ** 2)
    return loss
