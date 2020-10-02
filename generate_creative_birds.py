# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import cv2
import torch
import numpy as np
import argparse
import torchvision
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from retry.api import retry_call
from torch.utils import data
from torchvision import transforms
from part_selector import Trainer as Trainer_selector
from part_generator import Trainer as Trainer_cond_unet
from scipy.ndimage.morphology import distance_transform_edt

COLORS = {'initial':1-torch.cuda.FloatTensor([45, 169, 145]).view(1, -1, 1, 1)/255., 'eye':1-torch.cuda.FloatTensor([243, 156, 18]).view(1, -1, 1, 1)/255., 'none':1-torch.cuda.FloatTensor([149, 165, 166]).view(1, -1, 1, 1)/255., 
        'beak':1-torch.cuda.FloatTensor([211, 84, 0]).view(1, -1, 1, 1)/255., 'body':1-torch.cuda.FloatTensor([41, 128, 185]).view(1, -1, 1, 1)/255., 'details':1-torch.cuda.FloatTensor([171, 190, 191]).view(1, -1, 1, 1)/255.,
        'head':1-torch.cuda.FloatTensor([192, 57, 43]).view(1, -1, 1, 1)/255., 'legs':1-torch.cuda.FloatTensor([142, 68, 173]).view(1, -1, 1, 1)/255., 'mouth':1-torch.cuda.FloatTensor([39, 174, 96]).view(1, -1, 1, 1)/255., 
        'tail':1-torch.cuda.FloatTensor([69, 85, 101]).view(1, -1, 1, 1)/255., 'wings':1-torch.cuda.FloatTensor([127, 140, 141]).view(1, -1, 1, 1)/255.}

class Initialstroke_Dataset(data.Dataset):
    def __init__(self, folder, image_size):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for p in Path(f'{folder}').glob(f'**/*.png')]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = self.transform(Image.open(path))
        return img

    def sample(self, n):
        sample_ids = [np.random.randint(self.__len__()) for _ in range(n)]
        samples = [self.transform(Image.open(self.paths[sample_id])) for sample_id in sample_ids]
        return torch.stack(samples).cuda()

def load_latest(model_dir, name):
    model_dir = Path(model_dir)
    file_paths = [p for p in Path(model_dir / name).glob('model_*.pt')]
    saved_nums = sorted(map(lambda x: int(x.stem.split('_')[1]), file_paths))
    if len(saved_nums) == 0:
        return
    name = saved_nums[-1]
    print(f'continuing from previous epoch - {name}')
    return name


def noise(n, latent_dim):
    return torch.randn(n, latent_dim).cuda()

def noise_list(n, layers, latent_dim):
    return [(noise(n, latent_dim), layers)]

def mixed_list(n, layers, latent_dim):
    tt = int(torch.rand(()).numpy() * layers)
    return noise_list(n, tt, latent_dim) + noise_list(n, layers - tt, latent_dim)

def image_noise(n, im_size):
    return torch.FloatTensor(n, im_size, im_size, 1).uniform_(0., 1.).cuda()

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

def gs_to_rgb(image, color):
    image_rgb = image.repeat(1, 3, 1, 1)
    return 1-image_rgb*color

@torch.no_grad()
def generate_truncated(S, G, style, noi, trunc_psi = 0.75, num_image_tiles = 8, bitmap_feats=None, batch_size=8):
    latent_dim = G.latent_dim
    z = noise(2000, latent_dim)
    samples = evaluate_in_chunks(batch_size, S, z).cpu().numpy()
    av = np.mean(samples, axis = 0)
    av = np.expand_dims(av, axis = 0)
        
    w_space = []
    for tensor, num_layers in style:
        tmp = S(tensor)
        av_torch = torch.from_numpy(av).cuda()
        # import ipdb;ipdb.set_trace()
        tmp = trunc_psi * (tmp - av_torch) + av_torch
        w_space.append((tmp, num_layers))

    w_styles = styles_def_to_tensor(w_space)
    generated_images = evaluate_in_chunks_unet(batch_size, G, bitmap_feats, w_styles, noi)
    return generated_images.clamp_(0., 1.)


@torch.no_grad()
def generate_part(model, partial_image, partial_rgb, color=None, percentage=20, num=0, num_image_tiles=8, trunc_psi=1., save_img=False, results_dir='../results', evolvement=False):
    model.eval()
    ext = 'png'
    num_rows = np.sqrt(num_image_tiles)
    latent_dim = model.G.latent_dim
    image_size = model.G.image_size
    num_layers = model.G.num_layers
    if percentage == 'eye':
        n_eye = 10
        generated_partial_images_candidates = []
        scores = torch.zeros(n_eye)
        for _ in range(n_eye):
            latents_z = noise_list(num_image_tiles, num_layers, latent_dim)
            n = image_noise(num_image_tiles, image_size)
            image_partial_batch = partial_image[:, -1:, :, :]
            bitmap_feats = model.Enc(partial_image)
            generated_partial_images = generate_truncated(model.S, model.G, latents_z, n, trunc_psi = trunc_psi, bitmap_feats=bitmap_feats)
            generated_partial_images_candidates.append(generated_partial_images)
        generated_partial_images_candidates = torch.cat(generated_partial_images_candidates, 0)
        # eye size rank
        n_pixels = generated_partial_images_candidates.sum(-1).sum(-1).sum(-1) # B
        for rank, i_eye in enumerate(torch.argsort(n_pixels, descending=True)):
            scores[i_eye] += (rank+1)/n_eye
        # eye distance rank
        initial_stroke = partial_image[:, :1].cpu().data.numpy()
        initial_stroke_dt = torch.cuda.FloatTensor(distance_transform_edt(1-initial_stroke))
        dt_pixels = (generated_partial_images_candidates*initial_stroke_dt).sum(-1).sum(-1).sum(-1) # B
        for rank, i_eye in enumerate(torch.argsort(dt_pixels, descending=False)): # the smaller the better
            if n_pixels[i_eye] > 3:
                scores[i_eye] += (rank+1)/n_eye
        generated_partial_images = generated_partial_images_candidates[torch.argsort(scores, descending=True)[0]].unsqueeze(0)
    else:
        # latents and noise
        latents_z = noise_list(num_image_tiles, num_layers, latent_dim)
        n = image_noise(num_image_tiles, image_size)
        image_partial_batch = partial_image[:, -1:, :, :]
        bitmap_feats = model.Enc(partial_image)
        generated_partial_images = generate_truncated(model.S, model.G, latents_z, n, trunc_psi = trunc_psi, bitmap_feats=bitmap_feats)
    # regular
    generated_partial_images = generate_truncated(model.S, model.G, latents_z, n, trunc_psi = trunc_psi, bitmap_feats=bitmap_feats)
    generated_partial_rgb = gs_to_rgb(generated_partial_images, color)
    generated_images = generated_partial_images + image_partial_batch
    generated_rgb = 1 - ((1-generated_partial_rgb)+(1-partial_rgb))
    if save_img:
        torchvision.utils.save_image(generated_partial_rgb, os.path.join(results_dir, f'{str(num)}-{percentage}-comp.{ext}'), nrow=num_rows)
        torchvision.utils.save_image(generated_rgb, os.path.join(results_dir, f'{str(num)}-{percentage}.{ext}'), nrow=num_rows)
    return generated_partial_images.clamp_(0., 1.), generated_images.clamp_(0., 1.), generated_partial_rgb.clamp_(0., 1.), generated_rgb.clamp_(0., 1.)
    

def train_from_folder(
    data_path = '../../data',
    results_dir = '../../results',
    models_dir = '../../models',
    n_part = 1,
    image_size = 128,
    network_capacity = 16,
    batch_size = 3,
    num_image_tiles = 8,
    trunc_psi = 0.75,
    generate_all=False,
):
    min_step = 299
    name_eye='short_bird_creative_sequential_r6_partstack_aug_eye_unet2_largeaug'
    load_from = load_latest(models_dir, name_eye)
    load_from = min(min_step, load_from)
    model_eye = Trainer_cond_unet(name_eye, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_eye.load_config()
    model_eye.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_eye, load_from)))

    name_head='short_bird_creative_sequential_r6_partstack_aug_head_unet2'
    load_from = load_latest(models_dir, name_head)
    load_from = min(min_step, load_from)
    model_head = Trainer_cond_unet(name_head, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_head.load_config()
    model_head.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_head, load_from)))
    
    name_body='short_bird_creative_sequential_r6_partstack_aug_body_unet2'
    load_from = load_latest(models_dir, name_body)
    load_from = min(min_step, load_from)
    model_body = Trainer_cond_unet(name_body, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_body.load_config()
    model_body.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_body, load_from)))
    
    name_beak='short_bird_creative_sequential_r6_partstack_aug_beak_unet2'
    load_from = load_latest(models_dir, name_beak)
    load_from = min(min_step, load_from)
    model_beak = Trainer_cond_unet(name_beak, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_beak.load_config()
    model_beak.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_beak, load_from)))
    
    name_legs='short_bird_creative_sequential_r6_partstack_aug_legs_unet2'
    load_from = load_latest(models_dir, name_legs)
    load_from = min(min_step, load_from)
    model_legs = Trainer_cond_unet(name_legs, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_legs.load_config()
    model_legs.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_legs, load_from)))
    
    name_wings='short_bird_creative_sequential_r6_partstack_aug_wings_unet2'
    load_from = load_latest(models_dir, name_wings)
    load_from = min(min_step, load_from)
    model_wings = Trainer_cond_unet(name_wings, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_wings.load_config()
    model_wings.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_wings, load_from)))
    
    name_mouth='short_bird_creative_sequential_r6_partstack_aug_mouth_unet2'
    load_from = load_latest(models_dir, name_mouth)
    load_from = min(min_step, load_from)
    model_mouth = Trainer_cond_unet(name_mouth, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_mouth.load_config()
    model_mouth.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_mouth, load_from)))
    
    name_tail='short_bird_creative_sequential_r6_partstack_aug_tail_unet2'
    load_from = load_latest(models_dir, name_tail)
    load_from = min(min_step, load_from)
    model_tail = Trainer_cond_unet(name_tail, results_dir,  models_dir, n_part=n_part, batch_size=batch_size, image_size=image_size, network_capacity=network_capacity)
    model_tail.load_config()
    model_tail.GAN.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_tail, load_from)))
    

    name_selector='short_bird_creative_selector_aug'
    load_from = load_latest(models_dir, name_selector)
    part_selector = Trainer_selector(name_selector, results_dir, models_dir, n_part=n_part, batch_size = batch_size, image_size = image_size, network_capacity = network_capacity)
    part_selector.load_config()
    part_selector.clf.load_state_dict(torch.load('%s/%s/model_%d.pt'%(models_dir, name_selector, load_from)))

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    inital_dir = '%s/bird_short_test_init_strokes_%d'%(data_path, image_size)
    dataset = Initialstroke_Dataset(inital_dir, image_size=image_size)
    dataloader = data.DataLoader(dataset, num_workers=5, batch_size=batch_size, drop_last=False, shuffle=False, pin_memory=True)
    # import ipdb;ipdb.set_trace()

    models = [model_eye, model_head, model_body, model_beak, model_legs, model_wings, model_mouth, model_tail]
    target_parts = ['eye', 'head', 'body', 'beak', 'legs', 'wings', 'mouth', 'tail', 'none']
    part_to_id = {'initial': 0, 'eye': 1, 'head': 4, 'body': 3, 'beak': 2, 'legs': 5, 'wings': 8, 'mouth': 6, 'tail': 7}
    max_iter = 10
    if generate_all:
        generation_dir = os.path.join(results_dir, 'DoodlerGAN_all')
        if not os.path.exists(generation_dir):
            os.mkdir(generation_dir)
            os.mkdir(os.path.join(generation_dir, 'bw'))
            os.mkdir(os.path.join(generation_dir, 'color_initial'))
            os.mkdir(os.path.join(generation_dir, 'color'))
        for count, initial_strokes in enumerate(dataloader):
            initial_strokes = initial_strokes.cuda()
            start_point = len(os.listdir(os.path.join(generation_dir, 'bw')))
            print('%d sketches generated'%start_point)
            for i in range(batch_size):
                samples_name = f'generated-{start_point+i}'
                stack_parts = torch.zeros(1, 10, image_size, image_size).cuda()
                initial_strokes_rgb = gs_to_rgb(initial_strokes[i], COLORS['initial'])
                stack_parts[:, 0] = initial_strokes[i, 0]
                stack_parts[:, -1] = initial_strokes[i, 0]
                partial_rgbs = initial_strokes_rgb.clone()
                prev_part = []
                for iter_i in range(max_iter):
                    outputs = part_selector.clf.D(stack_parts)
                    part_rgbs = torch.ones(1, 3, image_size, image_size).cuda()
                    select_part_order = 0
                    select_part_ids = torch.topk(outputs, k=8, dim=0)[1]
                    select_part_id = select_part_ids[select_part_order].item()
                    select_part = target_parts[select_part_id]
                    while (select_part == 'none' and iter_i < 6 or select_part in prev_part):
                        select_part_order += 1
                        if select_part_order > 7:
                            import ipdb;ipdb.set_trace()
                        select_part_id = select_part_ids[select_part_order].item()
                        select_part = target_parts[select_part_id]
                    if select_part == 'none':
                        break
                    prev_part.append(select_part)
                    sketch_rgb = partial_rgbs
                    stack_part = stack_parts.clone()
                    select_model = models[select_part_id]
                    part, partial, part_rgb, partial_rgb = generate_part(select_model.GAN, stack_part, sketch_rgb, COLORS[select_part], select_part, samples_name, 1, results_dir=results_dir, trunc_psi=0.1)
                    stack_parts[0, part_to_id[select_part]] = part[0, 0]
                    partial_rgbs[0] = partial_rgb[0]
                    stack_parts[0, -1] = partial[0, 0]
                    part_rgbs[0] = part_rgb[0]
                initial_colored_full = np.tile(np.max(stack_parts.cpu().data.numpy()[:, 1:-1], 1), [3, 1, 1])
                initial_colored_full = 1-np.max(np.stack([1-initial_strokes_rgb.cpu().data.numpy()[0], initial_colored_full]), 0)
                cv2.imwrite(os.path.join(generation_dir, 'bw', f'{str(samples_name)}.png'), (1-stack_parts[0, -1].cpu().data.numpy())*255.)
                cv2.imwrite(os.path.join(generation_dir, 'color_initial', f'{str(samples_name)}-color.png'), cv2.cvtColor(initial_colored_full.transpose(1, 2, 0)*255., cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(generation_dir, 'color', f'{str(samples_name)}-color.png'), cv2.cvtColor(partial_rgbs[0].cpu().data.numpy().transpose(1, 2, 0)*255., cv2.COLOR_RGB2BGR))
    else:
        now = datetime.now()
        timestamp = now.strftime("%m-%d-%Y_%H-%M-%S")
        stack_parts = torch.zeros(num_image_tiles*num_image_tiles, 10, image_size, image_size).cuda()
        initial_strokes = dataset.sample(num_image_tiles*num_image_tiles).cuda()
        initial_strokes_rgb = gs_to_rgb(initial_strokes, COLORS['initial'])
        stack_parts[:, 0] = initial_strokes[:, 0]
        stack_parts[:, -1] = initial_strokes[:, 0]
        partial_rgbs = initial_strokes_rgb.clone()
        partial_rgbs_variation = initial_strokes_rgb.clone()
        prev_parts = [[] for _ in range(num_image_tiles**2)]
        samples_name = f'generated-{timestamp}-{min_step}'
        for iter_i in range(max_iter):
            outputs = part_selector.clf.D(stack_parts)
            part_rgbs = torch.ones(num_image_tiles*num_image_tiles, 3, image_size, image_size).cuda()
            for i in range(num_image_tiles**2):
                prev_part = prev_parts[i]
                select_part_order = 0
                select_part_ids = torch.topk(outputs[i], k=9, dim=0)[1]
                select_part_id = select_part_ids[select_part_order].item()
                select_part = target_parts[select_part_id]
                while (select_part == 'none' and iter_i < 6 or select_part in prev_part):
                    select_part_order += 1
                    select_part_id = select_part_ids[select_part_order].item()
                    select_part = target_parts[select_part_id]
                if select_part == 'none':
                    break
                prev_parts[i].append(select_part)
                sketch_rgb = partial_rgbs[i].clone().unsqueeze(0)
                stack_part = stack_parts[i].unsqueeze(0)
                select_model = models[select_part_id]
                part, partial, part_rgb, partial_rgb = generate_part(select_model.GAN, stack_part, sketch_rgb, COLORS[select_part], select_part, samples_name, 1, results_dir=results_dir, trunc_psi=0.1)
                stack_parts[i, part_to_id[select_part]] = part[0, 0]
                stack_parts[i, -1] = partial[0, 0]
                partial_rgbs[i] = partial_rgb[0]
                part_rgbs[i] = part_rgb[0]
            torchvision.utils.save_image(partial_rgbs, os.path.join(results_dir, f'{str(samples_name)}-{str(min_step)}-round{iter_i}.png'), nrow=num_image_tiles)
            torchvision.utils.save_image(part_rgbs, os.path.join(results_dir, f'{str(samples_name)}-{str(min_step)}-part-round{iter_i}.png'), nrow=num_image_tiles)
        torchvision.utils.save_image(1-stack_parts[:, -1:], os.path.join(results_dir, f'{str(samples_name)}-{str(min_step)}-final_pred.png'), nrow=num_image_tiles)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='../data')
    parser.add_argument("--results_dir", type=str, default='../results/creative_bird_generation')
    parser.add_argument("--models_dir", type=str, default='../models')
    parser.add_argument('--n_part', type=int, default=10)
    parser.add_argument('--image_size', type=int, default=64)
    parser.add_argument('--network_capacity', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--num_image_tiles', type=int, default=8)
    parser.add_argument('--trunc_psi', type=float, default=1.)
    parser.add_argument('--generate_all', action='store_true')

    args = parser.parse_args()
    print(args)

    train_from_folder(args.data_dir, args.results_dir, args.models_dir, args.n_part, args.image_size, args.network_capacity, 
        args.batch_size, args.num_image_tiles, args.trunc_psi, args.generate_all)