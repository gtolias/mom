# Mining on Manifolds: Metric Learning Without Labels

This is a Matlab package for our paper:

> A. Iscen, G. Tolias, Y. Avrithis, O. Chum. "Mining on Manifolds: Metric Learning Without Labels", CVPR 2018

It implements unsupervised selection of training pairs (in MATLAB) and training for fine-grained categorization (in Python/PyTorch).


## Prerequisites

1. [Package](https://github.com/ahmetius/diffusion-retrieval) for diffusion proposed in our CVPR17 paper: 

> A. Iscen, G. Tolias, Y. Avrithis, T. Furon, O. Chum. "Efficient Diffusion on Region Manifolds: Recovering Small Objects with Compact CNN Representations", CVPR 2017

If not available, it is automatically downloaded within the main script. 

2. MatConvNet:

It is used to extract descriptor for training images using a pre-trained network. This will be the input to the mining process. The code is tested with MatConvNet version 1.0-beta25 in MATLAB R2016b.

3. Pytorch: 

It is used to implement the training for fine-grained categorization on the CUB-200-2011 dataset. The code is tested with PyTorch version 1.0.1.post2 and Python 2.7.13 on Debian 8.1.


## Execution

### Demo on 1M web-crawled images

We provide the initial descriptors for 1 million images (particular object retrieval experiment in CVPR18), perform training pair selection with MoM, and qualitatively present results using image thumbnails.

Run the following script through MATLAB:
```
>> run mat/mom_1M
```

### Fine-grained categorization

Training, in the form of metric learning, is performed for fine-grained bird categorization. The initial descriptor extraction and selection by MoM is performed in MATLAB. The training, based on an earlier version of [this package](https://github.com/vadimkantorov/metriclearningbench), is then performed in Python/Pytorch.

Download CUB_200_2011 with:

```
python py/download_dataset.py
```

Extract descriptors for training images by running through MATLAB:

```
>> run mat/extract_descriptors_CUB
```

Perform data selection for training by running through MATLAB:

```
>> run mat/mom_CUB
```

Perform the training with:
```
>> python py/train.py
```
