import pdb
import os, random, tarfile
import numpy as np
from PIL import Image
from skimage import io, transform

import torch, torchvision
import torch.utils.data as data
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR10
from torchvision.datasets.utils import check_integrity, download_url


# =====================================================================================
class CUB2011():
 root = 'data/'
 url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
 filename = 'CUB_200_2011.tgz'
 tgz_md5 = '97eceeb196236b17998738112f37df78'

 num_training_classes = 100
 name = 'CUB_200_2011'
 triplet_mode = False
 mined_data = None

 integrity_test_list = [
  ['001.Black_footed_Albatross/Black_Footed_Albatross_0001_796111.jpg', '4c84da568f89519f84640c54b7fba7c2'],
  ['002.Laysan_Albatross/Laysan_Albatross_0001_545.jpg', 'e7db63424d0e384dba02aacaf298cdc0'],
  ['198.Rock_Wren/Rock_Wren_0001_189289.jpg', '487d082f1fbd58faa7b08aa5ede3cc00'],
  ['200.Common_Yellowthroat/Common_Yellowthroat_0003_190521.jpg', '96fd60ce4b4805e64368efc32bf5c6fe']
 ]


 def __init__(self, transform=None, download=False, train = True, **kwargs):
  if download and not _check_integrity(self.root+'/CUB_200_2011/images/', self.integrity_test_list):
   download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

  if not _check_integrity(self.root+'/CUB_200_2011/images/', self.integrity_test_list):
   raise RuntimeError('Dataset not found or corrupted. You can use download=True to download it')

  self.transform = transform
  self.train = train
 
  self.classes = [x.split()[-1] for x in open(self.root+self.name+"/classes.txt", "r").readlines()]
  self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
  self.classes = self.classes[:self.num_training_classes] if train else self.classes[self.num_training_classes:]
 
  self.idx_to_class = {v: k for k, v in self.class_to_idx.iteritems()}

  i = open(self.root+self.name+"/images.txt", "r").readlines()
  l = open(self.root+self.name+"/image_class_labels.txt", "r").readlines()
  self.imgs = [(_[0].split()[1], int(_[1].split()[1])-1) for _ in zip(i,l)]
  self.imgs = [(self.root+"/"+self.name+"/images/"+image_file_path, class_label_ind) for image_file_path, class_label_ind in self.imgs if ((class_label_ind-self.num_training_classes)<0) == self.train ]

  self.loader = pil_loader

 def __getitem__(self, index):

  # triplet (anchor, pos, neg) mode
  if self.triplet_mode:
   perm = random.sample(self.mined_data['anchors'].keys(), 1)[0] # pick a random anchor

   # sort negatives based on Euclidean distance to anchor
   q_x = self.embeddings[perm,:]
   n_x = self.embeddings[self.mined_data['negpool'][perm],:]
   dists = ((q_x - n_x)**2).sum(axis=1)
   n_idx = np.argsort(dists) 
   
   rand_pos = np.random.randint(0, len(self.mined_data['pospool'][perm])) # random pos from the pool
   rand_neg = n_idx[np.random.randint(0, np.min((10, len(self.mined_data['negpool'][perm]))))]  # random pick from 10 Euclidean-NN in the pool

   a_path, a_target = self.imgs[int(self.mined_data['anchors'][perm])]
   p_path, p_target = self.imgs[int(self.mined_data['pospool'][perm][rand_pos])]
   n_path, n_target = self.imgs[int(self.mined_data['negpool'][perm][rand_neg])]

   p_w = self.mined_data['posweight'][perm][rand_pos]
   n_w = self.mined_data['negweight'][perm][rand_neg]

   a_img, p_img, n_img = self.loader(a_path), self.loader(p_path), self.loader(n_path) 

   if self.transform is not None:
    a_img, p_img, n_img = self.transform(a_img), self.transform(p_img), self.transform(n_img)

   return a_img, p_img, n_img, p_w, n_w

  # single image mode
  else:
   path, target = self.imgs[index]
   img = self.loader(path)
   if self.transform is not None:
    img = self.transform(img)

   return img, target


 def __len__(self):
  return len(self.imgs)

# =====================================================================================
# taken from different pytorch version
def download_and_extract_archive(url, download_root, extract_root=None, filename=None,
                                 md5=None, remove_finished=False):
    download_root = os.path.expanduser(download_root)
    if extract_root is None:
        extract_root = download_root
    if not filename:
        filename = os.path.basename(url)

    download_url(url, download_root, filename, md5)

    archive = os.path.join(download_root, filename)
    print("Extracting {} to {}".format(archive, extract_root))
    
    with tarfile.open(archive, 'r:gz') as tar: tar.extractall(path=extract_root)


# =====================================================================================
# taken from different pytorch version
def _check_integrity(img_folder, integrity_test_list):
 for fentry in (integrity_test_list):
  filename, md5 = fentry[0], fentry[1]
  fpath = img_folder + filename
  if not check_integrity(fpath, md5):
      return False
 return True


# =====================================================================================
def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

