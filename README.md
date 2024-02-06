# Introduction

this project is to predict attribute of molecule with GNN

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## create dataset

```shell
python3 create_dataset.py --input_csv <path/to/train/csv> --output_tfrecord trainset.tfrecord
python3 create_dataset.py --input_csv <path/to/test/csv> --output_tfrecord testset.tfrecord
```

## train

```shell
python3 train.py --dataset dataset
```
