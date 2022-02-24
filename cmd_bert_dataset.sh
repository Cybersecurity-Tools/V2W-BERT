#!/bin/bash

/bin/hostname

module load anaconda/2020.11
conda activate deep_38 

cd /homes/das90/GNNcodes/GNN-NC/V2W-BERT/

python PrepareDataset.py --dir='Dataset' --from_year=2020 --to_year=2021 --from_train_year=1990 --to_train_year=2020 --from_test_year=2021 --to_test_year=2021 --from_val_year=2022 --to_test_year=2022

