# Creative Sketch Generation (DoodlerGAN)

DoodlerGAN is a part-based Generative Adversarial Network (GAN) designed to generate creative sketches with unseen compositions of novel part appearances. Concretely, DoodlerGAN contains two modules: the part generator and the part selector. Given a part-based representation of a partial sketch, the part selector predicts which part category to draw next. Given a part-based representation of a partial sketch and a part category, the part generator generates a raster image of the part (which represents both the appearance and location of the part). Some randomly selected generation with DoodlerGAN trained on Creative Birds and Creative Creatures dataset are shown below.

![Generated Sketches](figs/generation.png)

## Preparation

First, create the enviroment with Anaconda. Installing Pytorch and the other packages listed as requirements.txt. The code is tested with PyTorch 1.3.1 and CUDA 10.0:

```
  mkdir creative_sketch_generation creative_sketch_generation/data creative_sketch_generation/results creative_sketch_generation/models
  cd creative_sketch_generation
  git clone git@github.com:fairinternal/AI-doodler.git
  conda create -n doodler python=3.7
  conda activate doodler
  conda install pytorch==1.3.1 -c pytorch
  pip install -r requirements.txt
```

Then download our processed Creative Birds and Creative Creatures datasets from the GoogleDrive: https://drive.google.com/drive/folders/14ZywlSE-khagmSz23KKFbLCQLoMOxPzl?usp=sharing and unzip them under the directory `creative_sketch_generation/data/`.

## Usage

### Training

Refer to the `training_scripts` folder for the scripts that reproduce our results. Example usages of training part generators and part selectors are shown below:

```
python run_part_generator.py --new --results_dir ../results --models_dir ../models --n_part 10 --data ../data/ird_short_wings_json_64 --name short_bird_creative_wings --num_train_steps 300000 --batch_size 40 --network_capacity 16 --gradient_accumulate_every 1 --save_every 2000 --image_size 64 --sparsity_penalty 0.01 --learning_rate_D 1e-4 --learning_rate_G 1e-4
python run_part_generator.py --new --results_dir ../results --models_dir ../models --large_aug --n_part 19 --data ../data/generic_long_legs_json_64 --name long_generic_creative_legs --batch_size 40 --gradient_accumulate_every 1 --network_capacity 16 --save_every 2000 --image_size 64 --sparsity_penalty 0.01 --learning_rate_D 1e-4 --learning_rate_G 1e-4 --num_train_steps 600000
python run_part_selector.py --new --results_dir ../results --models_dir ../models --n_part 10 --data ../data/bird_short_ --name short_bird_creative_selector --batch_size 128 --save_every 1000 --image_size 64
```

### Inference

The part generators and part selector are used iteratively to complete the entire sketche given random initial strokes during the inference. To generate a `[num_image_tiles x num_image_tiles]` grid to visualize the generations based on the trained model, one can use the following scripts. We also release our trained models on the GoogleDrive.

```
python generate_creative_birds.py --models_dir ../models --results_dir ../results/creative_bird_generation --data_dir ../data --num_image_tiles 8
python generate_creative_creatures.py --models_dir ../models --results_dir ../results/creative_creature_generation --data_dir ../data --num_image_tiles 10
```

To generate 10,000 sketches for quantitative evaluation, use `--generate_all` flag as below. The script will automatically create three folders under `results_dir`: `DoodlerGAN_all/bw`, `DoodlerGAN_all/color`, and `DoodlerGAN_all/color_initial`, which include the generations in grayscale, or with different parts or colored initial stroke colored.

```
python generate_creative_birds.py --generate_all --models_dir ../models --results_dir ../results/creative_bird_generation --data_dir ../data --num_image_tiles 8
python generate_creative_creatures.py --generate_all --models_dir ../models --results_dir ../results/creative_creature_generation --data_dir ../data --num_image_tiles 10
```

### Quantitative Evaluation

We analyze the quality and novelty of the generations with four metrics: Frechet inception distances (FID), generation diversity (GD), characteristic score (CS) and semantic diversity score (SDS). Please refer to our papers for more details of the metrics. To run the evaluation, use the following script with indicated generator directory and real image directory:

```
python evaluate.py training_dir generation_dir --gpu 1 --name birds
```

## License

DoodlerGAN is MIT licensed, as found in the LICENSE file.
