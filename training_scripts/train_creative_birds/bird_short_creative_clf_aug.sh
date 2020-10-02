# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=short_bird_creative_selector
#SBATCH --output=../../../jobs/sample-short_bird_creative_selector-%j.out
#SBATCH --error=../../../jobs/sample-short_bird_creative_selector-%j.err
#SBATCH --partition=short
#SBATCH --time=40:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

python ../../run_part_selector.py --new --results_dir ../../../results  --models_dir ../../../models --n_part 10 --data ../../../data/bird_short_ --name short_bird_creative_selector --batch_size 128 --save_every 1000 --image_size 64
