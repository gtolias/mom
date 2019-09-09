import os, sys, random, argparse, hickle, pdb
import numpy as np
from itertools import izip

import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.optim import lr_scheduler

import cub2011 as cub2011
import inception_v1_googlenet
import model


# get embeddings for all images in the dataset
def get_dataset_embeddings(model, dataset, threads = 8): 
 embeddings_all, labels_all = [], []
 loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers = threads)
 for batch_idx, batch in enumerate(loader):
  with torch.no_grad():
   images, labels = [torch.autograd.Variable(tensor.cuda()) for tensor in batch]
  embeddings_all.extend(model(images).data.cpu().numpy())
  labels_all.extend(labels.data.cpu().numpy())
 
 return np.asarray(embeddings_all), np.asarray(labels_all)

# nn search and recall computation
def recall(embeddings, labels, K = 1):
 prod = torch.mm(embeddings, embeddings.t())
 norm = prod.diag().unsqueeze(1).expand_as(prod)
 D = norm + norm.t() - 2 * prod
 knn_inds = D.topk(1 + K, dim = 1, largest = False)[1][:, 1:]
 return (labels.unsqueeze(-1).expand_as(knn_inds) == labels[knn_inds.contiguous().view(-1)].view_as(knn_inds)).max(1)[0].float().mean()

# load MoM result from text files
def read_mined_data(anchors_fn, pos_fn, neg_fn, posw_fn, negw_fn):

 anchors, pos, neg, posw, negw = dict(), dict(), dict(), dict(), dict()
 with open(anchors_fn) as f:
     for idx,line in enumerate(f):
         anchors[idx] = int(line.strip())-1

 with open(pos_fn) as posf, open(neg_fn) as negf, open(posw_fn) as poswf, open(negw_fn) as negwf:
     for idx, (pos_line, neg_line, posw_line, negw_line) in enumerate(izip(posf, negf, poswf, negwf)):
         pos[idx] = [x-1 for x in map(int,pos_line.strip().split(','))]
         neg[idx] = [x-1 for x in map(int,neg_line.strip().split(','))]
         posw[idx] = map(float,posw_line.strip().split(','))
         negw[idx] = map(float,negw_line.strip().split(','))

 return {'anchors':anchors,'pospool':pos,'negpool':neg,'posweight':posw,'negweight':negw}


parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__ = lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--epochs', default = 100, type = int)
parser.add_argument('--batch', default = 42, type = int)
parser.add_argument('--lr', default = 0.01, type = float)
parser.add_argument('--margin', default = 0.5, type = float)
parser.add_argument('--step-size', default = 50, type = int)
parser.add_argument('--gamma', default = 0.1, type = float)
parser.add_argument('--gpu-id', default='0', type=str)
opts = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id

emb_dim = 64
  
data_dir = 'data/' 
log = open(data_dir+'log.txt', 'a', 0)

# set random seeds
for set_random_seed in [random.seed, np.random.seed, torch.manual_seed, torch.cuda.manual_seed_all]: set_random_seed(8)

# load base model
base_model = inception_v1_googlenet.inception_v1_googlenet()
base_model_weights_path = os.path.join(data_dir+'inception_v1_googlenet.h5')
base_model.load_state_dict({k : torch.from_numpy(v) for k, v in hickle.load(base_model_weights_path).items()})

normalize = transforms.Compose([
 transforms.ToTensor(),
 transforms.Lambda(lambda x: x * base_model.rescale),
 transforms.Normalize(mean = base_model.rgb_mean, std = base_model.rgb_std),
 transforms.Lambda(lambda x: x[[2, 1, 0], ...])
])

dataset_train = cub2011.CUB2011(train = True, transform = transforms.Compose([
 transforms.RandomResizedCrop(base_model.input_side),
 transforms.RandomHorizontalFlip(),
 normalize
]), download = True)

dataset_eval = cub2011.CUB2011(train = False, transform = transforms.Compose([
 transforms.Resize(256),
 transforms.CenterCrop(base_model.input_side),
 normalize
]), download = True)

# load training data picked by MoM
dataset_train.mined_data = read_mined_data(data_dir+'anchors.txt',data_dir+'pos.txt',data_dir+'neg.txt',data_dir+'posw.txt',data_dir+'negw.txt')

# loss, optimizer, scheduler
model = model.WeightedTriplet(base_model, dataset_train.num_training_classes, lr =opts.lr, embedding_size = emb_dim).cuda()
optimizer = torch.optim.SGD(model.parameters(), weight_decay = 5e-4, lr = opts.lr, momentum = 0.9, dampening = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **dict(step_size = opts.step_size, gamma = opts.gamma))

# evaluate on test set for initial network
model.eval()
embeddings_all, labels_all = get_dataset_embeddings(model,dataset_eval)
rec = [recall(torch.Tensor(embeddings_all), torch.Tensor(labels_all), x).item() for x in [1,2,4,8]]
print('recall@1,2,4,8 epoch {}: {:.06f} {:.06f} {:.06f} {:.06f}'.format(-1, rec[0], rec[1], rec[2], rec[3]))
log.write('recall@1,2,4,8 epoch {}: {:.06f} {:.06f} {:.06f} {:.06f}'.format(-1, rec[0], rec[1], rec[2], rec[3]))
 
for epoch in range(opts.epochs):

 # get embeddings and create train loader
 model.eval()
 dataset_train.triplet_mode = False
 dataset_train.embeddings, _ = get_dataset_embeddings(model,dataset_train)
 dataset_train.triplet_mode = True
 loader_train = torch.utils.data.DataLoader(dataset_train, num_workers = 8, batch_size = opts.batch, drop_last = True)
 model.train()

 # batch train
 scheduler.step()
 loss_all = []
 for batch_idx, batch in enumerate(loader_train if model.criterion is not None else []):  
  a_images, p_images, n_images, p_w, n_w  = [torch.autograd.Variable(tensor.cuda()) for tensor in batch]
  loss = model.criterion(model(a_images), model(p_images), model(n_images), p_w, n_w, margin = opts.margin)
  loss_all.append(loss.data.item())
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
 print('loss epoch {}: {:.04f}'.format(epoch, np.mean(loss_all)))
 log.write('loss epoch {}: {:.04f}\n'.format(epoch, np.mean(loss_all)))

 # evaluate on test set 
 if epoch < 10 or (epoch + 1) % 5 == 0 or (epoch + 1) == opts.epochs:
  model.eval()
  embeddings_all, labels_all = get_dataset_embeddings(model,dataset_eval)
  rec = [recall(torch.Tensor(embeddings_all), torch.Tensor(labels_all), x).item() for x in [1,2,4,8]]
  print('recall@1,2,4,8 epoch {}: {:.06f} {:.06f} {:.06f} {:.06f}'.format(epoch, rec[0], rec[1], rec[2], rec[3]))
  log.write('recall@1,2,4,8 epoch {}: {:.06f} {:.06f} {:.06f} {:.06f}'.format(epoch, rec[0], rec[1], rec[2], rec[3]))

torch.save({'epoch': epoch + 1, 'state_dict': model.state_dict()},'{}/checkpoint_{}.pth'.format(data_dir, epoch))
