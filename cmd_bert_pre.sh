#!/bin/bash

/bin/hostname

module load anaconda/2020.11
conda activate deep_38 

cd /homes/das90/GNNcodes/GNN-NC/V2W-BERT/

python V2WBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200

## Other dataset

#python V2WBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='dummy'

#python V2WBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='temporal'

#python V2WBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='random'


## To run in distributed dataparallel mode
#python V2WBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='ddp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='random'
