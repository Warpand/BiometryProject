# Biometry Project - Face Recognition

## Setup
### Environment
Install torch 2.6.0 and torchvision 0.21.0 with your desired cuda version, for example:
```shell
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```
Then run:
```shell
pip install -r requirements.txt
```
### Dataset
Download the [Casia-Webface](https://www.kaggle.com/datasets/ntl0601/casia-webface)
dataset, and place it in the [dataset](./dataset) directory (or create a symlink).
## Usage
### Training
Run:
```shell
python3 train.py
```
to start training.
You can overwrite the hyperparameters configuration using the cli (provided by
[tyro](https://brentyi.github.io/tyro/)).

Run:
```shell
python3 train.py --help
```
to see the possible options.
### Testing
Run:
```shell
python3 test.py -C PATH_TO_CHECKPOINT -r RESNET_VERSION -t THRESHOLD
```
to evaluate the model.

Run:
```shell
python3 test.py --help
```
to see additional options.
### Utilities
Casia-Webface is a dirty dataset and contains pixelated images.
```python3 preprocess_data.py``` was used to automatically filter them out. The
result is the [dataset/metadata.csv](./dataset/metadata.csv) file.

Checkpoints can be downloaded from wandb to local using:
```shell
python3 download_checkpoint.py -A PATH_TO_MODEL_ARTIFACT
```
