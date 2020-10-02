# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import json
from math import floor, log2
import random
from shutil import rmtree
from functools import partial
import multiprocessing

import numpy as np
import torch
from torch import nn
from torch.utils import data
import torch.nn.functional as F

from torch.optim import Adam

import torchvision

from PIL import Image
from pathlib import Path
import cairocffi as cairo

assert torch.cuda.is_available(), 'You need to have an Nvidia GPU with CUDA installed.'

num_cores = multiprocessing.cpu_count()

# helper classes

class NanException(Exception):
    pass

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)

# helpers

def default(value, d):
    return d if value is None else value

def cycle(iterable):
    while True:
        for i in iterable:
            yield i

def is_empty(t):
    return t.nelement() == 0

def raise_if_nan(t):
    if torch.isnan(t):
        raise NanException

def loss_backwards(loss, optimizer, **kwargs):
    loss.backward(**kwargs)

def leaky_relu(p=0.2):
    return nn.LeakyReLU(p, inplace=True)


class Dataset_JSON(data.Dataset):
    def __init__(self, base_path, name, image_size):
        super().__init__()
        self.image_size = image_size
        if 'bird' in name:
            self.target_parts = ['eye', 'head', 'body', 'beak', 'legs', 'wings', 'mouth', 'tail', 'none']
            self.id_to_part = {0:'initial', 1:'eye', 4:'head', 3:'body', 2:'beak', 5:'legs', 8:'wings', 6:'mouth', 7:'tail'}
        elif 'generic' in name or 'fin' in name or 'horn' in name:
            self.target_parts = ['eye', 'arms', 'beak', 'mouth', 'body', 'ears', 'feet', 'fin', 
                            'hair', 'hands', 'head', 'horns', 'legs', 'nose', 'paws', 'tail', 'wings', 'none']
            self.id_to_part = { 0:'initial',  1:'eye',  2:'arms',  3:'beak',  4:'mouth',  5:'body',  6:'ears',  7:'feet',  8:'fin', 
                             9:'hair',  10:'hands',  11:'head',  12:'horns',  13:'legs',  14:'nose',  15:'paws',  16:'tail', 17:'wings'}
        folder = base_path+'%s_json_'+'%d_train'%image_size
        self.paths = []
        self.paths_test = []
        # split the training data based on thte aids of the eye sketches
        for i, p in enumerate(Path(f'{folder%self.target_parts[0]}').glob(f'**/*.json')):
            if i%5 == 0:
                self.paths_test.append(p)
            else:
                self.paths.append(p)
        for part in self.target_parts[1:]:
            for i, p in enumerate(Path(f'{folder%part}').glob(f'**/*.json')):
                if Path(str(p).replace('_'+part, '_'+self.target_parts[0])) in self.paths_test:
                    self.paths_test.append(p)
                else:
                    self.paths.append(p)
        self.parts_id = [self.target_parts.index(str(path).split('_')[-5]) for path in self.paths]
        self.parts_id_test = [self.target_parts.index(str(path).split('_')[-5]) for path in self.paths_test]
        self.rotate = [-1/12*np.pi, 1/12*np.pi]
        self.trans = 0.01
        self.scale = [0.9, 1.1]
        self.n_part = len(self.id_to_part)

        self.samples_partid_test = [torch.LongTensor([self.parts_id_test[sample_id]]) for sample_id in range(self.__len_test__())]
        self.samples_partial_test = []
        for sample_id in range(self.__len_test__()):
            input_parts_json = json.load(open(self.paths_test[sample_id]))['input_parts']
            img_partial_test = []
            vector_input_part = []
            for i in range(self.n_part):
                key = self.id_to_part[i]
                vector_input_part += input_parts_json[key]
                img_partial_test.append(self.processed_part_to_raster(input_parts_json[key], side=self.image_size))
            img_partial_test.append(self.processed_part_to_raster(vector_input_part, side=self.image_size))
            self.samples_partial_test.append(torch.cat(img_partial_test, 0))
        # import ipdb;ipdb.set_trace()

        self.samples_partid_test = torch.stack(self.samples_partid_test)
        self.samples_partial_test = torch.stack(self.samples_partial_test)
        print(' | '.join(['%s : %d'%(target_part, (self.samples_partid_test==i).sum()) for i, target_part in enumerate(self.target_parts)])+
                ' | overall : %d'%(len(self.samples_partid_test)))

    def __len__(self):
        return len(self.paths)

    def __len_test__(self):
        return len(self.paths_test)

    def __getitem__(self, index):
        path = self.paths[index]
        part_id = self.parts_id[index]
        json_data = json.load(open(path))
        input_parts_json = json_data['input_parts']
        img_partial_test = []
        vector_input_part = []
        for i in range(self.n_part):
            key = self.id_to_part[i]
            vector_input_part += input_parts_json[key]
            img_partial_test.append(self.processed_part_to_raster(input_parts_json[key], side=self.image_size))
        img_partial_test.append(self.processed_part_to_raster(vector_input_part, side=self.image_size))
        # random affine
        theta = np.random.uniform(*self.rotate)
        trans_pixel = 512*self.trans
        translate_x = np.random.uniform(-trans_pixel, trans_pixel)
        translate_y = np.random.uniform(-trans_pixel, trans_pixel)
        scale = np.random.uniform(self.scale)
        # apply
        processed_img_partial = []
        affine_vector_input_part = []
        for i in range(self.n_part):
            key = self.id_to_part[i]
            affine_input_part_json = self.affine_trans(input_parts_json[key], theta, translate_x, translate_y, scale)
            affine_vector_input_part += affine_input_part_json
            processed_img_partial.append(self.processed_part_to_raster(affine_input_part_json, side=self.image_size))
        processed_img_partial.append(self.processed_part_to_raster(affine_vector_input_part, side=self.image_size))
        return part_id, torch.cat(processed_img_partial, 0), torch.cat(img_partial_test, 0)

    def sample_partial_test(self, n):
        sample_ids = [np.random.randint(self.__len__()) for _ in range(n)]
        samples_partid = [torch.LongTensor([self.parts_id[sample_id]]) for sample_id in sample_ids]
        sample_jsons = [json.load(open(self.paths[sample_id]))for sample_id in sample_ids]
        samples_partial = []
        for sample_json in sample_jsons:
            input_parts_json = sample_json['input_parts']
            img_partial_test = []
            vector_input_part = []
            for i in range(self.n_part):
                key = self.id_to_part[i]
                vector_input_part += input_parts_json[key]
                img_partial_test.append(self.processed_part_to_raster(input_parts_json[key], side=self.image_size))
            img_partial_test.append(self.processed_part_to_raster(vector_input_part, side=self.image_size))
            samples_partial.append(torch.cat(img_partial_test, 0))
        return torch.stack(samples_partid), torch.stack(samples_partial)

    def affine_trans(self, data, theta, translate_x, translate_y, scale):
        rotate_mat = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        affine_data = []
        for item in data:
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


class ClassifierBlock(nn.Module):
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


class Classifier(nn.Module):
    def __init__(self, image_size, network_capacity=16, n_part=1):
        super().__init__()
        num_layers = int(log2(image_size) - 1)
        num_init_filters = n_part

        blocks = []
        filters = [num_init_filters] + [(network_capacity) * (2 ** i) for i in range(num_layers+1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))

        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            num_layer = ind + 1
            is_not_last = ind < (len(chan_in_out) - 1)
            block = ClassifierBlock(in_chan, out_chan, downsample = is_not_last)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        latent_dim = 2 * 2 * filters[-1]

        self.flatten = Flatten()
        self.to_logit = nn.Linear(latent_dim, n_part-1)

    def forward(self, x):
        b, *_ = x.shape

        for block in self.blocks:
            x = block(x)

        x = self.flatten(x)
        x = self.to_logit(x)
        return x.squeeze()


class part_selector(nn.Module):
    def __init__(self, image_size, n_part=10, network_capacity=16, steps=1, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.steps = steps
        self.ema_decay = 0.995
        self.D = Classifier(image_size, network_capacity,n_part=n_part)
        self.D_opt = Adam(self.D.parameters(), lr = self.lr, betas=(0.5, 0.9))

        self._init_weights()

        self.cuda()
        
    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        return x

class Trainer():
    def __init__(self, name, results_dir, models_dir, n_part, image_size, network_capacity, batch_size = 4, 
                gradient_accumulate_every=1, lr = 2e-4, num_workers = None, save_every = 1000):
        self.clf = None

        self.name = name
        self.results_dir = Path(results_dir)
        self.models_dir = Path(models_dir)
        self.config_path = self.models_dir / name / '.config.json'

        assert log2(image_size).is_integer(), 'image size must be a power of 2 (64, 128, 256, 512, 1024)'
        self.n_part = n_part
        self.image_size = image_size
        self.network_capacity = network_capacity

        self.lr = lr
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.save_every = save_every
        self.steps = 0

        self.gradient_accumulate_every = gradient_accumulate_every

        self.d_loss = 0
        self.d_acc = 0

        self.loader = None

        self.criterion = nn.CrossEntropyLoss()

        if 'bird' in name:
            self.target_parts = ['eye', 'head', 'body', 'beak', 'legs', 'wing', 'mouth', 'tail', 'none']
        elif 'generic' in name or 'fin' in name or 'horn' in name:
            self.target_parts = ['eye', 'arms', 'beak', 'mouth', 'body', 'ears', 'feet', 'fin', 
                            'hair', 'hands', 'head', 'horns', 'legs', 'nose', 'paws', 'tail', 'wings', 'none']
        self.n_part_class = len(self.target_parts)

    def init_clf(self):
        self.clf = part_selector(n_part=self.n_part, lr=self.lr, image_size=self.image_size, network_capacity=self.network_capacity)

    def write_config(self):
        self.config_path.write_text(json.dumps(self.config()))

    def load_config(self):
        config = self.config() if not self.config_path.exists() else json.loads(self.config_path.read_text())
        self.image_size = config['image_size']
        self.network_capacity = config['network_capacity']
        del self.clf
        self.init_clf()

    def config(self):
        return {'image_size': self.image_size, 'network_capacity': self.network_capacity}

    def set_data_src(self, folder, name):
        self.dataset = Dataset_JSON(folder, name, self.image_size)
        print('Number of data: %d'%(len(self.dataset)))
        self.loader = cycle(data.DataLoader(self.dataset, num_workers=default(self.num_workers, num_cores), batch_size=self.batch_size, drop_last=True, shuffle=True, pin_memory=True))

    def train(self):
        self.init_folders()
        if self.clf is None:
            self.init_clf()

        self.clf.train()
        total_disc_loss = torch.tensor(0.).cuda()
        total_acc = torch.tensor(0.).cuda()
        batch_size = self.batch_size

        backwards = partial(loss_backwards)

        self.clf.D_opt.zero_grad()

        for i in range(self.gradient_accumulate_every):
            part_id_batch, image_cond_batch, _ = [item.cuda() for item in next(self.loader)]
            outputs = self.clf.D(image_cond_batch)
            _, predicts = torch.max(outputs, 1)
            acc = (predicts == part_id_batch).sum().float() / part_id_batch.size(0) / self.gradient_accumulate_every
            disc_loss = self.criterion(outputs, part_id_batch)
            disc_loss = disc_loss / self.gradient_accumulate_every
            disc_loss.register_hook(raise_if_nan)
            backwards(disc_loss, self.clf.D_opt)
            total_disc_loss += disc_loss.detach().item()
            total_acc += acc.detach().item()

        self.d_loss = float(total_disc_loss)
        self.d_acc = float(total_acc)
        self.clf.D_opt.step()

        # save from NaN errors

        checkpoint_num = floor(self.steps / self.save_every)

        if torch.isnan(total_disc_loss):
            print(f'NaN detected. Loading from checkpoint #{checkpoint_num}')
            self.load(checkpoint_num)
            raise NanException

        # periodically save results
        if self.steps % self.save_every == 0:
            self.save(checkpoint_num)

        if self.steps % 1000 == 0 or (self.steps % 100 == 0 and self.steps < 2500):
            self.evaluate(floor(self.steps / 1000))

        self.steps += 1

    @torch.no_grad()
    def evaluate(self, num = 0, num_image_tiles = 8):
        self.clf.eval()
        ext = 'png'
        num_rows = num_image_tiles
        part_id_batch, image_cond_batch = [item.cuda() for item in self.dataset.sample_partial_test(num_rows ** 2)]
        outputs = self.clf.D(image_cond_batch.clone().detach())
        _, predicted = torch.max(outputs, 1)
        with open(str(self.results_dir / self.name / f'{str(num)}-pred.txt'), 'w') as fw:
            for i in range(num_rows):
                for j in range(num_rows):
                    fw.write('%s\t'%self.target_parts[predicted[i*num_rows+j]])
                fw.write('\n')
        with open(str(self.results_dir / self.name / f'{str(num)}-real.txt'), 'w') as fw:
            for i in range(num_rows):
                for j in range(num_rows):
                    fw.write('%s\t'%self.target_parts[part_id_batch[i*num_rows+j]])
                fw.write('\n')
        torchvision.utils.save_image(image_cond_batch[:, -1:], str(self.results_dir / self.name / f'{str(num)}.{ext}'), nrow=num_rows)
        part_id_test, image_cond_test = self.dataset.samples_partid_test.cuda(), self.dataset.samples_partial_test.cuda()
        class_correct = list(0. for i in range(self.n_part_class))
        class_total = list(0. for i in range(self.n_part_class))
        n_batch = self.dataset.__len_test__()//256
        for i in range(n_batch+1):
            if i == n_batch:
                part_id_batch, image_cond_batch = part_id_test[i*256:], image_cond_test[i*256:]
            else:
                part_id_batch, image_cond_batch = part_id_test[i*256:(i+1)*256], image_cond_test[i*256:(i+1)*256]
            outputs = self.clf.D(image_cond_batch.clone().detach())
            _, predicts = torch.max(outputs, 1)
            with torch.no_grad():
                for part_id, pred_id in zip(part_id_batch, predicts):
                    c = (part_id == pred_id).squeeze()
                    class_correct[part_id] += c
                    class_total[part_id] += 1
        print(' | '.join(['%s: %.2f'%(target_part, 100*class_correct[i]/(class_total[i]+1e-6)) for i, target_part in enumerate(self.target_parts)])+
                ' | overall : %.2f'%(100*sum(class_correct)/(sum(class_total)+1e-6)))

    def print_log(self):
        print(f'training loss: {self.d_loss:.2f} | training acc: {self.d_acc:.2f}')

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
        torch.save(self.clf.state_dict(), self.model_name(num))
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
        self.clf.load_state_dict(torch.load(self.model_name(name)))
