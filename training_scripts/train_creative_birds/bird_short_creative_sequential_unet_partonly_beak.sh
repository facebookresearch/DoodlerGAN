# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=short_bird_creative_beak_unet
#SBATCH --output=../../../jobs/sample-short_bird_creative_beak-%j.out
#SBATCH --error=../../../jobs/sample-short_bird_creative_beak-%j.err
#SBATCH --partition=short
#SBATCH --time=72:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

python ../../run_part_generator.py --new --results_dir ../../../results  --models_dir ../../../models --n_part 10 --data ../../../data/bird_short_beak_json_64 --name short_bird_creative_beak --batch_size 40 --network_capacity 16 --gradient_accumulate_every 1 --save_every 2000 --image_size 64 --sparsity_penalty 0.01 --learning_rate_D 1e-4 --learning_rate_G 1e-4 --num_train_steps 300000
