# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
import torch
import random
import torchvision
import multiprocessing
import numpy as np
import torch.nn.functional as F
from math import floor, log2
from shutil import rmtree
from functools import partial

from torch import nn
from torch.utils import data
from torch.optim import Adam
from torch.autograd import grad as torch_grad
from torchvision import transforms

from PIL import Image
from pathlib import Path
import cairocffi as cairo

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

COLORS_BIRD = {'initial':1-torch.cuda.FloatTensor([45, 169, 145]).view(1, -1, 1, 1)/255., 'eye':1-torch.cuda.FloatTensor([243, 156, 18]).view(1, -1, 1, 1)/255., 'none':1-torch.cuda.FloatTensor([149, 165, 166]).view(1, -1, 1, 1)/255., 
        'beak':1-torch.cuda.FloatTensor([211, 84, 0]).view(1, -1, 1, 1)/255., 'body':1-torch.cuda.FloatTensor([41, 128, 185]).view(1, -1, 1, 1)/255., 'details':1-torch.cuda.FloatTensor([171, 190, 191]).view(1, -1, 1, 1)/255.,
        'head':1-torch.cuda.FloatTensor([192, 57, 43]).view(1, -1, 1, 1)/255., 'legs':1-torch.cuda.FloatTensor([142, 68, 173]).view(1, -1, 1, 1)/255., 'mouth':1-torch.cuda.FloatTensor([39, 174, 96]).view(1, -1, 1, 1)/255., 
        'tail':1-torch.cuda.FloatTensor([69, 85, 101]).view(1, -1, 1, 1)/255., 'wings':1-torch.cuda.FloatTensor([127, 140, 141]).view(1, -1, 1, 1)/255.}

COLORS_GENERIC = {'initial':1-torch.cuda.FloatTensor([45, 169, 145]).view(1, -1, 1, 1)/255., 'eye':1-torch.cuda.FloatTensor([243, 156, 18]).view(1, -1, 1, 1)/255., 'none':1-torch.cuda.FloatTensor([149, 165, 166]).view(1, -1, 1, 1)/255., 
        'arms':1-torch.cuda.FloatTensor([211, 84, 0]).view(1, -1, 1, 1)/255., 'beak':1-torch.cuda.FloatTensor([41, 128, 185]).view(1, -1, 1, 1)/255., 'mouth':1-torch.cuda.FloatTensor([54, 153, 219]).view(1, -1, 1, 1)/255.,
        'body':1-torch.cuda.FloatTensor([192, 57, 43]).view(1, -1, 1, 1)/255., 'ears':1-torch.cuda.FloatTensor([142, 68, 173]).view(1, -1, 1, 1)/255., 'feet':1-torch.cuda.FloatTensor([39, 174, 96]).view(1, -1, 1, 1)/255., 
        'fin':1-torch.cuda.FloatTensor([69, 85, 101]).view(1, -1, 1, 1)/255., 'hair':1-torch.cuda.FloatTensor([127, 140, 141]).view(1, -1, 1, 1)/255., 'hands':1-torch.cuda.FloatTensor([45, 63, 81]).view(1, -1, 1, 1)/255.,
        'head':1-torch.cuda.FloatTensor([241, 197, 17]).view(1, -1, 1, 1)/255., 'horns':1-torch.cuda.FloatTensor([51, 205, 117]).view(1, -1, 1, 1)/255., 'legs':1-torch.cuda.FloatTensor([232, 135, 50]).view(1, -1, 1, 1)/255., 
        'nose':1-torch.cuda.FloatTensor([233, 90, 75]).view(1, -1, 1, 1)/255., 'paws':1-torch.cuda.FloatTensor([160, 98, 186]).view(1, -1, 1, 1)/255., 'tail':1-torch.cuda.FloatTensor([58, 78, 99]).view(1, -1, 1, 1)/255., 
        'wings':1-torch.cuda.FloatTensor([198, 203, 207]).view(1, -1, 1, 1)/255., 'details':1-torch.cuda.FloatTensor([171, 190, 191]).view(1, -1, 1, 1)/255.}

num_cores = multiprocessing.cpu_count()

# constants

EXTS = ['jpg', 'png', 'npy']
EPS = 1e-8

# helper classes

class NanException(Exception):
    pass

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

# helpers
def gs_to_rgb(image, color):
    image_rgb = image.repeat(1, 3, 1, 1)
    return 1-image_rgb*color

def default(value, d):
    return d if value is None else value

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def is_empty(t):
    return t.nelement() == 1

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def loss_backwards(loss, optimizer, **kwargs):
    loss.backward(**kwargs)

def gradient_penalty(images, output, weight = 10):
    batch_size = images.shape[0]
    gradients = torch_grad(outputs=output, inputs=images,
                           grad_outputs=torch.ones(output.size()).cuda(),
                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(batch_size, -1)
    return weight * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

def calc_pl_lengths(styles, images):
    num_pixels = images.shape[2] * images.shape[3]
    pl_noise = torch.randn(images.shape).cuda() / math.sqrt(num_pixels)
    outputs = (images * pl_noise).sum()

    pl_grads = torch_grad(outputs=outputs, inputs=styles,
                          grad_outputs=torch.ones(outputs.shape).cuda(),
                          create_graph=True, retain_graph=True, only_inputs=True)[0]
    return (pl_grads ** 2).sum(dim=2).mean(dim=1).sqrt()

def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()

def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def latent_to_w(style_vectorizer, latent_descr):
    return [(style_vectorizer(z), num_layers) for z, num_layers in latent_descr]

def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)

def evaluate_in_chunks(max_batch_size, model, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    chunked_outputs = [model(*i) for i in split_args]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def evaluate_in_chunks_unet(max_batch_size, model, map_feats, *args):
    split_args = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), args))))
    split_map_feats = list(zip(*list(map(lambda x: x.split(max_batch_size, dim=0), map_feats))))
    chunked_outputs = [model(*i, j) for i, j in zip(split_args, split_map_feats)]
    if len(chunked_outputs) == 1:
        return chunked_outputs[0]
    return torch.cat(chunked_outputs, dim=0)

def styles_def_to_tensor(styles_def):
    return torch.cat([t[:, None, :].expand(-1, n, -1) for t, n in styles_def], dim=1)


# dataset

class Dataset_JSON(data.Dataset):
    def __init__(self, folder, image_size, large_aug=False):
        super().__init__()
        min_sample_num = 10000
        self.folder = folder
        self.image_size = image_size
        self.large_aug = large_aug
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.json')]
        while len(self.paths) < min_sample_num:
            self.paths.extend(self.paths)
        # notice the real influence of the trans / scale is side / 512 (original side) because of scalling in rendering
        if not large_aug:
            self.rotate = [-1/12*np.pi, 1/12*np.pi]
            self.trans = 0.01
            self.scale = [0.9, 1.1]
        else:
            self.rotate = [-1/4*np.pi, 1/4*np.pi]
            self.trans = 0.05
            self.scale = [0.75, 1.25]
            self.line_diameter_scale = [0.25, 1.25]
        if 'bird' in folder:
            self.id_to_part = {0:'initial', 1:'eye', 4:'head', 3:'body', 2:'beak', 5:'legs', 8:'wings', 6:'mouth', 7:'tail'}
        elif 'generic' in folder or 'fin' in folder or 'horn' in folder:
            self.id_to_part = { 0:'initial',  1:'eye',  2:'arms',  3:'beak',  4:'mouth',  5:'body',  6:'ears',  7:'feet',  8:'fin', 
                         9:'hair',  10:'hands',  11:'head',  12:'horns',  13:'legs',  14:'nose',  15:'paws',  16:'tail', 17:'wings'}
        self.n_part = len(self.id_to_part)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        json_data = json.load(open(path))
        input_parts_json = json_data['input_parts']
        target_part_json = json_data['target_part']
        # sample random affine parameters
        theta = np.random.uniform(*self.rotate)
        trans_pixel = 512*self.trans
        translate_x = np.random.uniform(-trans_pixel, trans_pixel)
        translate_y = np.random.uniform(-trans_pixel, trans_pixel)
        scale = np.random.uniform(*self.scale)
        if self.large_aug:
            line_diameter = np.random.uniform(*self.line_diameter_scale)*16
        else:
            line_diameter = 16
        # apply random affine transformation
        affine_target_part_json= self.affine_trans(target_part_json, theta, translate_x, translate_y, scale)
        processed_img_partial = []
        affine_vector_input_part = []
        for i in range(self.n_part):
            key = self.id_to_part[i]
            affine_input_part_json = self.affine_trans(input_parts_json[key], theta, translate_x, translate_y, scale)
            affine_vector_input_part += affine_input_part_json
            processed_img_partial.append(self.processed_part_to_raster(affine_input_part_json, side=self.image_size, line_diameter=line_diameter))
        processed_img_partial.append(self.processed_part_to_raster(affine_vector_input_part, side=self.image_size, line_diameter=line_diameter))
        processed_img_partonly = self.processed_part_to_raster(affine_target_part_json, side=self.image_size, line_diameter=line_diameter)
        processed_img = self.processed_part_to_raster(affine_vector_input_part+affine_target_part_json, side=self.image_size, line_diameter=line_diameter)
        # RandomHorizontalFlip
        if np.random.random() > 0.5:
            processed_img = processed_img.flip(-1)
            processed_img_partial = torch.cat(processed_img_partial, 0).flip(-1)
            processed_img_partonly = processed_img_partonly.flip(-1)
        else:
            processed_img_partial = torch.cat(processed_img_partial, 0)
        return processed_img, processed_img_partial, processed_img_partonly

    def sample_partial_test(self, n):
        sample_ids = [np.random.randint(self.__len__()) for _ in range(n)]
        sample_jsons = [json.load(open(self.paths[sample_id]))for sample_id in sample_ids]
        samples = []
        samples_partial = []
        samples_partonly = []
        for sample_json in sample_jsons:
            input_parts_json = sample_json['input_parts']
            target_part_json = sample_json['target_part']
            img_partial_test = []
            vector_input_part = []
            for i in range(self.n_part):
                key = self.id_to_part[i]
                vector_input_part += input_parts_json[key]
                img_partial_test.append(self.processed_part_to_raster(input_parts_json[key], side=self.image_size))
            img_partial_test.append(self.processed_part_to_raster(vector_input_part, side=self.image_size))
            samples_partial.append(torch.cat(img_partial_test, 0))
            img_partonly_test = self.processed_part_to_raster(target_part_json, side=self.image_size)
            img_test = self.processed_part_to_raster(vector_input_part+target_part_json, side=self.image_size)
            samples.append(img_test)
            samples_partonly.append(img_partonly_test)
        return torch.stack(samples), torch.stack(samples_partial), torch.stack(samples_partonly)

    def affine_trans(self, data, theta, translate_x, translate_y, scale):
        rotate_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        affine_data = []
        for item in data:
            if len(item) == 0:
                continue
            affine_item = np.array(item) - 256.
            affine_item = np.transpose(np.matmul(rotate_mat, np.transpose(affine_item)))
            affine_item[:, 0] += translate_x
            affine_item[:, 1] += translate_y
            affine_item *= scale
            affine_data.append(affine_item + 256.)
        return affine_data

    def processed_part_to_raster(self, vector_part, side=64, line_diameter=16, padding=16, bg_color=(0,0,0), fg_color=(1,1,1)):
        """
        render raster image based on the processed part
        """
        original_side = 512.
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, side, side)
        ctx = cairo.Context(surface)
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.set_line_width(line_diameter)
        # scale to match the new size
        # add padding at the edges for the line_diameter
        # and add additional padding to account for antialiasing
        total_padding = padding * 2. + line_diameter
        new_scale = float(side) / float(original_side + total_padding)
        ctx.scale(new_scale, new_scale)
        ctx.translate(total_padding / 2., total_padding / 2.)
        raster_images = []
        # clear background
        ctx.set_source_rgb(*bg_color)
        ctx.paint()
        # draw strokes, this is the most cpu-intensive part
        ctx.set_source_rgb(*fg_color)
        for stroke in vector_part:
            if len(stroke) == 0:
                continue
            ctx.move_to(stroke[0][0], stroke[0][1])
            for x, y in stroke:
                ctx.line_to(x, y)
            ctx.stroke()
        surface_data = surface.get_data()
        raster_image = np.copy(np.asarray(surface_data))[::4].reshape(side, side)
        return torch.FloatTensor(raster_image/255.)[None, :, :]

# exponential moving average helpers

def ema_inplace(moving_avg, new, decay):
    if is_empty(moving_avg):
        moving_avg.data.copy_(new)
        return
    moving_avg.data.mul_(decay).add_(1 - decay, new)


# Encoder

class EncoderBlock_unet(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        x = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

class Encoder_unet(nn.Module):
    def __init__(self, num_init_filters, image_size, network_capacity=16):
        super().__init__()
        num_layers = int(log2(image_size) - 1)

        blocks = []
        filters = [num_init_filters] + [network_capacity*(2 ** (i)) for i in range(num_layers)] # 16, 32, 64, 128, 256, 512, 1024
        chan_in_out = list(zip(filters[0:-1], filters[1:]))

        for ind, (in_chan, out_chan) in enumerate(chan_in_out): # 128, 512, 2048, 4096, 16384, 65536, 262144
            is_not_last = ind < (len(chan_in_out) - 1)
            block = EncoderBlock_unet(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        feats = []
        for block in self.blocks:
            x = block(x)
            feats.append(x)        
        return feats

# stylegan2_cond_unet classes
class StyleVectorizer(nn.Module):
    def __init__(self, emb, depth):
        super().__init__()

        layers = []
        for i in range(depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu()])

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, rgba = False):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)

        # out_filters = 3 if not rgba else 4
        out_filters = 1
        self.conv = Conv2DMod(input_channel, out_filters, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor = 2, mode='bilinear', align_corners=False) if upsample else None

    def forward(self, x, prev_rgb, istyle):
        b, c, h, w = x.shape
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1, **kwargs):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdim=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x

class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample = True, upsample_rgb = True, rgba = False):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)
        
        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu()
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, rgba)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb

class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu()
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding = 1, stride = 2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = x + res
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class Generator_unet(nn.Module):
    def __init__(self, image_size, latent_dim, network_capacity=16):
        super().__init__()
        self.image_size = image_size
        self.latent_dim = latent_dim
        self.num_layers = int(log2(image_size) - 1)

        init_channels = 4 * network_capacity
        self.initial_block = nn.Parameter(torch.randn((init_channels, 4, 4)))
        filters = [init_channels] + [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        in_out_pairs = zip([ch+network_capacity*(2 ** (self.num_layers-1-i)) for i, ch in enumerate(filters[0:-1])], filters[1:])

        self.blocks = nn.ModuleList([])
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                latent_dim,
                in_chan,
                out_chan,
                upsample = not_first,
                upsample_rgb = not_last
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise, cond_feat_maps):
        batch_size = styles.shape[0]
        image_size = self.image_size
        x = self.initial_block.expand(batch_size, -1, -1, -1)
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block, feat_map in zip(styles, self.blocks, cond_feat_maps[::-1]):
            x = torch.cat([x, feat_map], 1)
            x, rgb = block(x, rgb, style, input_noise)
        return rgb

class Discriminator(nn.Module):
    def __init__(self, image_size, network_capacity=16, n_part=1):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = n_part

        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers+1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))
        blocks = []

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind < (len(chan_in_out) - 1)
            block = DiscriminatorBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        latent_dim = 2 * 2 * filters[-1]

        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, 1)

    def forward(self, x):
        b, *_ = x.shape

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze()

class StyleGAN2_cond_unet(nn.Module):
    def __init__(self, image_size, n_part=10, latent_dim=512, style_depth=8, network_capacity=16, steps=1, lr_D=1e-4, lr_G=1e-4):
        super().__init__()
        self.lr_D = lr_D
        self.lr_G = lr_G
        self.steps = steps
        self.ema_decay = 0.995

        self.S = StyleVectorizer(latent_dim, style_depth)
        self.G = Generator_unet(image_size, latent_dim, network_capacity)
        self.D = Discriminator(image_size, network_capacity, n_part=n_part)
        self.Enc = Encoder_unet(n_part, image_size, network_capacity)

        self.generator_params = list(self.G.parameters()) + list(self.S.parameters()) + list(self.Enc.parameters())
        self.G_opt = Adam(self.generator_params, lr = self.lr_G, betas=(0., 0.99))
        self.D_opt = Adam(self.D.parameters(), lr = self.lr_D, betas=(0., 0.99))

        self._init_weights()
        self.cuda()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def forward(self, x):
        return x

class Trainer():
    def __init__(self, name, results_dir, models_dir, n_part, image_size, network_capacity, batch_size = 4, mixed_prob = 0.9, 
                gradient_accumulate_every=1, lr_D = 2e-4, lr_G = 2e-4, num_workers = None, save_every = 1000, trunc_psi = 0.6, sparsity_penalty=0.):
        self.GAN = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.n_part = n_part
        self.image_size = image_size
        self.network_capacity = network_capacity

        self.lr_D = lr_D
        self.lr_G = lr_G
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixed_prob = mixed_prob
        self.sparsity_penalty = sparsity_penalty

        self.save_every = save_every
        self.steps = 0

        self.trunc_psi = trunc_psi

        self.gradient_accumulate_every = gradient_accumulate_every

        self.d_loss = 0
        self.g_loss = 0
        self.last_gp_loss = 0
        self.pl_loss = 0
        self.sparsity_loss = 0

        self.pl_mean = torch.empty(1).cuda()
        self.pl_ema_decay = 0.99

        self.loader_D = None
        self.loader_G = None
        self.av = None

        if 'bird' in self.name:
            self.part_to_id = {'initial': 0, 'eye': 1, 'head': 4, 'body': 3, 'beak': 2, 'legs': 5, 'wings': 8, 'mouth': 6, 'tail': 7}
            COLORS = COLORS_BIRD
        elif 'generic' in self.name or 'fin' in self.name or 'horn' in self.name:
            self.part_to_id = {'initial': 0, 'eye': 1, 'arms': 2, 'beak': 3, 'mouth': 4, 'body': 5, 'ears': 6, 'feet': 7, 'fin': 8, 
                            'hair': 9, 'hands': 10, 'head': 11, 'horns': 12, 'legs': 13, 'nose': 14, 'paws': 15, 'tail': 16, 'wings':17}
            COLORS = COLORS_GENERIC

        self.color = 1-torch.cuda.FloatTensor([0, 0, 0]).view(1, -1, 1, 1)
        self.default_color = 1-torch.cuda.FloatTensor([0, 0, 0]).view(1, -1, 1, 1)
        for key in COLORS:
            if key in self.name:
                self.color = COLORS[key]
                break

        for partname in self.part_to_id.keys():
            if partname in self.name:
                self.partid = self.part_to_id[partname]
                self.partname = partname

    def init_GAN(self):
        self.GAN = StyleGAN2_cond_unet(n_part=self.n_part,  lr_G=self.lr_G,  lr_D=self.lr_D, image_size = self.image_size, network_capacity = self.network_capacity)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        del self.GAN
        self.init_GAN()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def set_data_src(self, folder, large_aug=False):
        self.dataset_D = Dataset_JSON(folder, self.image_size, large_aug=large_aug)
        self.dataset_G = Dataset_JSON(folder, self.image_size, large_aug=large_aug)
        self.loader_D = cycle(data.DataLoader(self.dataset_D, num_workers = default(self.num_workers, num_cores), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))
        self.loader_G = cycle(data.DataLoader(self.dataset_G, num_workers = default(self.num_workers, num_cores), batch_size = self.batch_size, drop_last = True, shuffle=True, pin_memory=True))

    def train(self):
        assert self.loader_G is not None, 'You must first initialize the data source with `.set_data_src(<folder of images>)`'

        self.init_folders()

        if self.GAN is None:
            self.init_GAN()

        self.GAN.train()
        total_disc_loss = torch.tensor(0.).cuda()
        total_gen_loss = torch.tensor(0.).cuda()

        batch_size = self.batch_size

        image_size = self.GAN.G.image_size
        latent_dim = self.GAN.G.latent_dim
        num_layers = self.GAN.G.num_layers

        apply_gradient_penalty = self.steps % 4 == 0
        apply_path_penalty = self.steps % 32 == 0

        backwards = partial(loss_backwards)

        avg_pl_length = self.pl_mean
        self.GAN.D_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in next(self.loader_D)]
            image_partial_batch = image_cond_batch[:, -1:, :, :] # take the first one as the entire input partial sketch
            get_latents_fn = mixed_list if np.random.random() < self.mixed_prob else noise_list
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            bitmap_feats = self.GAN.Enc(image_cond_batch)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_partial_images = self.GAN.G(w_styles, noise, bitmap_feats)
            generated_images = torch.max(generated_partial_images, image_partial_batch)

            generated_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(generated_partial_images, image_cond_batch[:, self.partid:self.partid+1]),
                                                    image_cond_batch[:, self.partid+1:-1], generated_images], 1)
            fake_output = self.GAN.D(generated_image_stack_batch.clone().detach())

            image_batch.requires_grad_()
            real_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(part_only_batch, image_cond_batch[:, self.partid:self.partid+1]),
                                                    image_cond_batch[:, self.partid+1:-1], image_batch], 1)
            real_image_stack_batch.requires_grad_()
            real_output = self.GAN.D(real_image_stack_batch)

            disc_loss = (F.relu(1 + real_output) + F.relu(1 - fake_output)).mean()

            if apply_gradient_penalty:
                gp = gradient_penalty(real_image_stack_batch, real_output)
                self.last_gp_loss = gp.clone().detach().item()
                disc_loss = disc_loss + gp

            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.GAN.D_opt)

            total_disc_loss += disc_loss.detach().item() / self.gradient_accumulate_every

        self.d_loss = float(total_disc_loss)
        self.GAN.D_opt.step()

        # train generator

        self.GAN.G_opt.zero_grad()
        for i in range(self.gradient_accumulate_every):
            image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in next(self.loader_G)]
            image_partial_batch = image_cond_batch[:, -1:, :, :] # take the first one as the entire input partial sketch
            
            style = get_latents_fn(batch_size, num_layers, latent_dim)
            noise = image_noise(batch_size, image_size)

            bitmap_feats = self.GAN.Enc(image_cond_batch)

            w_space = latent_to_w(self.GAN.S, style)
            w_styles = styles_def_to_tensor(w_space)

            generated_partial_images = self.GAN.G(w_styles, noise, bitmap_feats)
            generated_images = torch.max(generated_partial_images, image_partial_batch)
            
            generated_image_stack_batch = torch.cat([image_cond_batch[:, :self.partid], torch.max(generated_partial_images, image_cond_batch[:, self.partid:self.partid+1]),
                                                    image_cond_batch[:, self.partid+1:-1], generated_images], 1)
            fake_output = self.GAN.D(generated_image_stack_batch)

            loss = fake_output.mean()
            gen_loss = loss

            if apply_path_penalty:
                pl_lengths = calc_pl_lengths(w_styles, generated_images)
                avg_pl_length = pl_lengths.detach().mean()

                if not is_empty(self.pl_mean):
                    pl_loss = ((pl_lengths - self.pl_mean) ** 2).mean()
                    if not torch.isnan(pl_loss):
                        gen_loss = gen_loss + pl_loss
                        if self.similarity_penalty:
                            gen_loss = gen_loss - self.similarity_penalty*(pl_lengths ** 2).mean()

            if self.sparsity_penalty:
                generated_density = generated_partial_images.reshape(self.batch_size, -1).sum(1)
                target_density = part_only_batch.reshape(self.batch_size, -1).sum(1) # if we devide the sketch by parts
                self.sparsity_loss = ((generated_density-target_density)**2).mean()
                gen_loss = gen_loss + self.sparsity_loss*self.sparsity_penalty

            gen_loss = gen_loss / self.gradient_accumulate_every
            gen_loss.register_hook(raise_if_nan)
            backwards(gen_loss, self.GAN.G_opt)

            total_gen_loss += loss.detach().item() / self.gradient_accumulate_every

        self.g_loss = float(total_gen_loss)
        self.GAN.G_opt.step()

        # calculate moving averages

        if apply_path_penalty and not torch.isnan(avg_pl_length):
            ema_inplace(self.pl_mean, avg_pl_length, self.pl_ema_decay)
            self.pl_loss = self.pl_mean.item()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if any(torch.isnan(l) for l in (total_gen_loss, total_disc_loss)):
            print(f'NaN detected for generator or discriminator. Loading from checkpoint #{checkpoint_num}')
            self.load(checkpoint_num)
            raise NanException

        # periodically save results

        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1
        self.av = None

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8, trunc = 1.0, rgb = False):
        self.GAN.eval()
        ext = 'png'
        num_rows = num_image_tiles
    
        # latent_dim = self.GAN.G.latent_dim - self.GAN.Enc.feat_dim
        latent_dim = self.GAN.G.latent_dim
        image_size = self.GAN.G.image_size
        num_layers = self.GAN.G.num_layers

        # latents and noise

        latents_z = noise_list(num_rows ** 2, num_layers, latent_dim)
        n = image_noise(num_rows ** 2, image_size)

        image_batch, image_cond_batch, part_only_batch = [item.cuda() for item in self.dataset_G.sample_partial_test(num_rows ** 2)]
        image_partial_batch = image_cond_batch[:, -1:, :, :] # take the first one as the entire input partial sketch

        # concat the two latent vectors
        bitmap_feats = self.GAN.Enc(image_cond_batch)

        generated_partial_images = self.generate_truncated(self.GAN.S, self.GAN.G, latents_z, n, trunc_psi = self.trunc_psi, bitmap_feats=bitmap_feats)
        generated_images = torch.max(generated_partial_images, image_partial_batch)
        
        if not rgb:
            torchvision.utils.save_image(image_partial_batch, str(self.results_dir / self.name / f'{str(num)}-part.{ext}'), nrow=num_rows)
            # torchvision.utils.save_image((image_batch-image_partial_batch).clamp_(0., 1.), str(self.results_dir / self.name / f'{str(num)}-real.{ext}'), nrow=num_rows)
            torchvision.utils.save_image(part_only_batch, str(self.results_dir / self.name / f'{str(num)}-real.{ext}'), nrow=num_rows)
            torchvision.utils.save_image(image_batch, str(self.results_dir / self.name / f'{str(num)}-full.{ext}'), nrow=num_rows)
            # regular
            torchvision.utils.save_image(generated_partial_images, str(self.results_dir / self.name / f'{str(num)}-comp.{ext}'), nrow=num_rows)
            torchvision.utils.save_image(generated_images.clamp_(0., 1.), str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        else:
            # part_batch = (image_batch-image_partial_batch).clamp_(0., 1.)
            partial_rgb = gs_to_rgb(image_partial_batch, self.default_color)
            # part_rgb = gs_to_rgb(part_batch, self.color)
            part_rgb = gs_to_rgb(part_only_batch, self.color)
            torchvision.utils.save_image(partial_rgb, str(self.results_dir / self.name / f'{str(num)}-part.{ext}'), nrow=num_rows)
            torchvision.utils.save_image(part_rgb, str(self.results_dir / self.name / f'{str(num)}-real.{ext}'), nrow=num_rows)
            torchvision.utils.save_image(1-((1-part_rgb)+(1-partial_rgb).clamp_(0., 1.)), str(self.results_dir / self.name / f'{str(num)}-full.{ext}'), nrow=num_rows)
            # regular
            generated_part_rgb = gs_to_rgb(generated_partial_images, self.color)
            torchvision.utils.save_image(generated_part_rgb, str(self.results_dir / self.name / f'{str(num)}-comp.{ext}'), nrow=num_rows)
            torchvision.utils.save_image(1-((1-generated_part_rgb)+(1-partial_rgb).clamp_(0., 1.)), str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)

    @torch.no_grad()
    def generate_truncated(self, S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8, bitmap_feats=None):
        latent_dim = G.latent_dim

        if self.av is None:
            z = noise(2000, latent_dim)
            samples = evaluate_in_chunks(self.batch_size, S, z).cpu().numpy()
            self.av = np.mean(samples, axis = 0)
            self.av = np.expand_dims(self.av, axis = 0)
            
        w_space = []
        for tensor, num_layers in style:
            tmp = S(tensor)
            av_torch = torch.from_numpy(self.av).cuda()
            tmp = trunc_psi * (tmp - av_torch) + av_torch
            w_space.append((tmp, num_layers))

        w_styles = styles_def_to_tensor(w_space)
        generated_images = evaluate_in_chunks_unet(self.batch_size, G, bitmap_feats, w_styles, noi)
        return generated_images.clamp_(0., 1.)

    def print_log(self):
        print(f'G: {self.g_loss:.2f} | D: {self.d_loss:.2f} | GP: {self.last_gp_loss:.2f} | PL: {self.pl_loss:.2f} | SP {self.sparsity_loss:.2f}')

    def model_name(self, num):
        return str(self.models_dir / self.name / f'model_{num}.pt')

    def init_folders(self):
        (self.results_dir / self.name).mkdir(parents=True, exist_ok=True)
        (self.models_dir / self.name).mkdir(parents=True, exist_ok=True)

    def clear(self):
        rmtree(str(self.models_dir / self.name), True)
        rmtree(str(self.results_dir / self.name), True)
        rmtree(str(self.config_path), True)
        self.init_folders()

    def save(self, num):
        torch.save(self.GAN.state_dict(), self.model_name(num))
        self.write_config()

    def load(self, num = -1):
        self.load_config()

        name = num
        if num == -1:
            file_paths = [p for p in Path(self.models_dir / self.name).glob('model_*.pt')]
            saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
            if len(saved_nums) == 0:
                return
            name = saved_nums[-1]
            print(f'continuing from previous epoch - {name}')
        self.steps = name * self.save_every
        self.GAN.load_state_dict(torch.load(self.model_name(name)))
