# Introduction

this project is to predict attribute of molecule with GNN

# Usage

## Install prerequisite

```shell
python3 -m pip install -r requirements.txt
```

## create dataset

```shell
python3 create_dataset.py --input_csv <path/to/csv> --output_dir dataset
```

## train

```shell
python3 train.py --dataset dataset
```
