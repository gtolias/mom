import torch
import torch.nn as nn
import torch.nn.functional as F

def distance_vectors_pairwise(anchor, positive, negative , squared = True):
    """Given batch of anchor descriptors and positive descriptors calculate distance matrix"""
    eps = 1e-8

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)
    n_sq = torch.sum(negative * negative, dim=1)

    d_a_p = a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1)
    d_a_n = a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1)
    d_p_n = p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1)

    if not squared:
     d_a_p = torch.sqrt(d_a_p + eps)
     d_a_n = torch.sqrt(d_a_n + eps)
     d_p_n = torch.sqrt(d_p_n + eps)

    return d_a_p, d_a_n, d_p_n

class Model(nn.Module):
 def __init__(self, base_model, num_classes, embedding_size = 128, lr = 0.001):
  super(Model, self).__init__()
  self.base_model = base_model
  self.num_classes = num_classes
  self.embedder = nn.Linear(base_model.output_size, embedding_size)
  self.lr = lr

 def forward(self, input):
  return self.embedder(F.relu(self.base_model(input).view(len(input), -1)))
 
 criterion = None

class WeightedTriplet(Model):
 def forward(self, input):
  return F.normalize(Model.forward(self, input))

 def criterion(self, a_emb, p_emb, n_emb, p_w, n_w, margin = 1.0):
  (d_a_p, d_a_n, _) = distance_vectors_pairwise(a_emb, p_emb, n_emb)
  loss = torch.clamp(margin + d_a_p - d_a_n, min=0.0)
  loss = loss * p_w.float()
  loss = torch.mean(loss)

  return loss