# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.



"""Calculates the Frechet Inception Distance (FID) to evalulate GANs

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectively.

See --help to see further details.

Code apapted from https://github.com/bioinf-jku/TTUR to use PyTorch instead
of Tensorflow

Copyright 2018 Institute of Bioinformatics, JKU Linz

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import cv2
import json
import pathlib
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision

import numpy as np
import torch
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
import torch.nn.functional as F

from PIL import Image

try:
    from tqdm import tqdm
except ImportError:
    # If not tqdm is not available, provide a mock version of it
    def tqdm(x): return x

from inception import InceptionV3

parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
parser.add_argument('path', type=str, nargs=2,
                    help=('Path to the generated images or '
                          'to .npz statistic files'))
parser.add_argument('--batch-size', type=int, default=50,
                    help='Batch size to use')
parser.add_argument('--dims', type=int, default=2048,
                    choices=list(InceptionV3.BLOCK_INDEX_BY_DIM),
                    help=('Dimensionality of Inception features to use. '
                          'By default, uses pool3 features'))
parser.add_argument('-c', '--gpu', default='', type=str,
                    help='GPU to use (leave blank for CPU only)')
parser.add_argument('--name', default='birds', type=str,
                    help='which dataset to be evluated', choices=['birds', 'creatures'])


with open('../data/id_to_class.json', 'r') as fp:
    ID2CLASS = json.load(fp)
    ID2CLASS ={int(k): v for k, v in ID2CLASS.items()}

B_SET = ['bird', 'duck', 'flamingo', 'parrot']
C_SET = ['ant', 'bear', 'bee', 'bird', 'butterfly', 'camel', 'cat', 'cow', 'crab', 'crocodile', 'dog', 'dolphin', 'duck', 
        'elephant', 'fish', 'flamingo', 'frog', 'giraffe', 'hedgehog', 'horse', 'kangaroo', 'lion', 'lobster', 'monkey', 'mosquito', 
        'mouse', 'octopus', 'owl', 'panda', 'parrot', 'penguin', 'pig', 'rabbit', 'raccoon', 'rhinoceros', 'scorpion', 'sea_turtle', 
        'shark', 'sheep', 'snail', 'snake', 'spider', 'squirrel', 'swan', 'tiger', 'whale', 'zebra']

def imread(filename):
    """
    Loads an image file into a (height, width, 3) uint8 ndarray.
    """
    return np.asarray(Image.open(filename), dtype=np.uint8)


def resize(sketch):
    x_nonzero, y_nonzero = np.where(sketch>0)
    try:
        coord_min = min(x_nonzero.min(), y_nonzero.min())
        coord_max = max(x_nonzero.max(), y_nonzero.max())
        sketch_new = np.zeros([64, 64])
        sketch_cropped = cv2.resize(sketch[coord_min:coord_max, coord_min:coord_max], (60, 60))
        sketch_new[2:-2, 2:-2] = sketch_cropped
    except:
        sketch_new = sketch
    return sketch_new

def resize_batch(sketches):
    return np.array([resize(sketch) for sketch in sketches])

def get_acts_and_preds(files, model, batch_size=50, dims=2048,
                    cuda=False, verbose=False, name='birds'):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : Batch size of images for the model to process at once.
                     Make sure that the number of samples is a multiple of
                     the batch size, otherwise some samples are ignored. This
                     behavior is retained to match the original FID score
                     implementation.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the number
                     of calculated batches is reported.
    -- name        : The name of the dataset: for birds we calculate CS only and for creatures we also calculate SDS.
    """
    if name == 'birds':
        target_set = B_SET
    elif name == 'creatures':
        target_set = C_SET

    model.eval()

    if batch_size > len(files):
        print(('Warning: batch size is bigger than the data size. '
               'Setting batch size to data size'))
        batch_size = len(files)

    pred_arr = np.empty((len(files), dims))
    preds_final_arr = {}
    logits_arr = torch.zeros(345).cuda()

    for i in tqdm(range(0, len(files), batch_size)):
        if verbose:
            print('\rPropagating batch %d/%d' % (i + 1, n_batches),
                  end='', flush=True)
        start = i
        end = i + batch_size

        images = np.array([imread(str(f)).astype(np.float32) for f in files[start:end]])
        images = images/255.
        images = 1-images
        images[images<0.1] = 0

        # Reshape to (n_images, 3, height, width)
        if len(images.shape) == 4:
            images = images.transpose((0, 3, 1, 2))
        elif len(images.shape) == 3:
            images = np.expand_dims(images, 1)

        batch = torch.from_numpy(images).type(torch.FloatTensor)
        if cuda:
            batch = batch.cuda()

        batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)

        # store the model predictions
        logits = model.inception(batch)
        _, final_preds = torch.max(logits, 1)
        logits = F.softmax(logits, 1)
        for logit, final_pred in zip(logits, final_preds):
            logits_arr += logit
            pred_class = ID2CLASS[final_pred.item()]
            if pred_class in preds_final_arr:
                preds_final_arr[pred_class] += 1
            else:
                preds_final_arr[pred_class] = 1

        
        pred = model(batch)[0]        

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred_arr[start:end] = pred.cpu().data.numpy().reshape(pred.size(0), -1)

    # calculate CS and SDS
    characteristic_count = 0.
    total_count = 0.
    for class_name in preds_final_arr:
        total_count += preds_final_arr[class_name]
        if class_name not in target_set:
            continue
        characteristic_count += preds_final_arr[class_name]
    CS = characteristic_count/total_count
    probs_all = logits_arr / total_count
    # import ipdb;ipdb.set_trace()
    if name == 'creatures':
        C_prob = sum([probs_all[cl_id].item() for cl_id in range(345) if ID2CLASS[cl_id] in C_SET])
        CCS = sum([-probs_all[cl_id].item()*np.log(probs_all[cl_id].item()/C_prob) for cl_id in range(345) if ID2CLASS[cl_id] in C_SET])
    else:
        CCS = 0.

    if verbose:
        print(' done')
    return pred_arr, CS, CCS


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)


def calculate_acts_and_preds(files, model, batch_size=50,
                                    dims=2048, cuda=False, verbose=False, name='birds'):
    """Calculation of the statistics used by the FID and diversity, CS, SDS.
    Params:
    -- files       : List of image files paths
    -- model       : Instance of inception model
    -- batch_size  : The images numpy array is split into batches with
                     batch size batch_size. A reasonable batch size
                     depends on the hardware.
    -- dims        : Dimensionality of features returned by Inception
    -- cuda        : If set to True, use GPU
    -- verbose     : If set to True and parameter out_step is given, the
                     number of calculated batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the inception model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the inception model.
    -- diversity : average pairwise distances between samples.
    -- CS : characteristic score.
    -- SDS : semantic diversity score.
    """
    assert name in ['birds', 'creatures']
    act, CS, SDS = get_acts_and_preds(files, model, batch_size, dims, cuda, verbose, name)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    diversity = cal_diversity(act)
    # import ipdb;ipdb.set_trace()
    return mu, sigma, diversity, CS, SDS


def cal_diversity(act):
    n_sample = min(act.shape[0], 1000)
    act = act[:n_sample]
    n_part = n_sample*(n_sample-1)/2
    score = 0.
    for i in range(n_sample):
        for j in range(i+1, n_sample):
            score += np.sqrt(np.sum((act[i]-act[j])**2))
    return score/n_part


def _compute_statistics_of_path(path, model, batch_size, dims, cuda, name):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        m, s, diversity, CS, SDS = calculate_acts_and_preds(files, model, batch_size,
                                               dims, cuda, False, name)
    return m, s, diversity, CS, SDS


def calculate_scores_given_paths(paths, batch_size, cuda, dims, name):
    """Calculates the FID of two paths"""
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx], normalize_input=False, use_fid_inception=False)
    if cuda:
        model.cuda()

    m1, s1, d1, CS1, SDS1 = _compute_statistics_of_path(paths[0], model, batch_size,
                                         dims, cuda, name)
    m2, s2, d2, CS2, SDS2 = _compute_statistics_of_path(paths[1], model, batch_size,
                                         dims, cuda, name)
    # import ipdb;ipdb.set_trace()
    fid_value = calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value, d1, d2, CS1, CS2, SDS1, SDS2


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    fid_value, d1, d2, CS1, CS2, SDS1, SDS2 = calculate_scores_given_paths(args.path,
                                          args.batch_size,
                                          args.gpu != '',
                                          args.dims,
                                          args.name)
    print('FID: ', fid_value)
    if args.name == 'birds':
        print('Diversity 1: %.2f, characteristic score 1: %.2f'%(d1, CS1))
        print('Diversity 2: %.2f, characteristic score 2: %.2f'%(d2, CS2))
    elif args.name == 'creatures':
        print('Diversity 1: %.2f, characteristic score 1: %.2f, semantic diversity score 1: %.2f'%(d1, CS1, SDS1))
        print('Diversity 2: %.2f, characteristic score 2: %.2f, semantic diversity score 2: %.2f'%(d2, CS2, SDS2))

