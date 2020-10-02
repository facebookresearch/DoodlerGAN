# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=long_generic_creative_selector
#SBATCH --output=../../../jobs/sample-long_generic_creative_selector_64_split_aug-%j.out
#SBATCH --error=../../../jobs/sample-long_generic_creative_selector_64_split_aug-%j.err
#SBATCH --partition=short
#SBATCH --gpus-per-node=1
#SBATCH --time=60:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10

python ../../run_part_selector.py --new --results_dir ../../../results  --models_dir ../../../models --n_part 19 --data ../../../data/generic_long_ --name long_generic_creative_selector_64 --batch_size 128 --save_every 1000 --image_size 64
