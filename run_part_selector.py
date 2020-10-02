# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from retry.api import retry_call
from tqdm import tqdm
from part_selector import Trainer, NanException
from datetime import datetime

def train_from_folder(
    data = '../../data',
    results_dir = '../../results',
    models_dir = '../../models',
    name = 'default',
    new = False,
    load_from = -1,
    n_part = 1,
    image_size = 128,
    network_capacity = 16,
    batch_size = 3,
    gradient_accumulate_every = 5,
    num_train_steps = 150000,
    learning_rate = 2e-4,
    num_workers =  None,
    save_every = 1000,
    num_image_tiles = 8,
):
    model = Trainer(
        name,        
        results_dir,
        models_dir,
        batch_size = batch_size,
        gradient_accumulate_every = gradient_accumulate_every,
        n_part = n_part,
        image_size = image_size,
        network_capacity = network_capacity,
        lr = learning_rate,
        num_workers = num_workers,
        save_every = save_every,
    )

    if not new:
        model.load(load_from)
    else:
        model.clear()

    model.set_data_src(data, name)

    for _ in tqdm(range(num_train_steps - model.steps), mininterval=10., desc=f'{name}<{data}>'):
        retry_call(model.train, tries=3, exceptions=NanException)
        if _ % 50 == 0:
            model.print_log()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default='../../data')
    parser.add_argument("--results_dir", type=str, default='../results')
    parser.add_argument("--models_dir", type=str, default='../models')
    parser.add_argument("--name", type=str, default='default')
    parser.add_argument("--load_from", type=int, default=-1)
    parser.add_argument('--new', action='store_true')
    parser.add_argument('--n_part', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=128)
    parser.add_argument('--network_capacity', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--gradient_accumulate_every', type=int, default=1)
    parser.add_argument('--num_train_steps', type=int, default=200000)
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--save_every', type=int, default=1000)
    parser.add_argument('--num_image_tiles', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)

    args = parser.parse_args()
    print(args)

    train_from_folder(args.data, args.results_dir, args.models_dir, args.name, args.new, args.load_from, args.n_part, 
        args.image_size, args.network_capacity, args.batch_size, args.gradient_accumulate_every, args.num_train_steps,  
        args.learning_rate, args.num_workers, args.save_every, args.num_image_tiles)
