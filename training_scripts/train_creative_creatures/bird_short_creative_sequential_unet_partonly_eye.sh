# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash
#SBATCH --job-name=long_generic_creative_eye_unet
#SBATCH --output=../../../jobs/sample-long_generic_creative_eye-%j.out
#SBATCH --error=../../../jobs/sample-long_generic_creative_eye-%j.err
#SBATCH --partition=short

#SBATCH --gpus-per-node=1
#SBATCH --time=72:00:00

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=20


python ../../run_part_generator.py --new --results_dir ../../../results  --models_dir ../../../models --large_aug --n_part 19 --data ../../../data/generic_long_eye_json_64_--partial ../../../data/generic_long_eye_input_parts_64_split_--name long_generic_creative_eye --batch_size 40 --gradient_accumulate_every 1 --network_capacity 16 --save_every 2000 --image_size 64 --sparsity_penalty 0.01 --learning_rate_D 1e-4 --learning_rate_G 1e-4 --num_train_steps 600000
