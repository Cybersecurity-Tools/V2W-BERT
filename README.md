<!-- https://gist.github.com/PurpleBooth/109311bb0361f32d87a2 -->
<!-- https://pandao.github.io/editor.md/en.html -->

# V2W-BERT

Instructions for V2W-BERT

### Dataset

Dataset: [https://drive.google.com/drive/folders/10E6nOXhRERhAmRVWla5i99jeUGWmI-Xl?usp=sharing](https://drive.google.com/drive/folders/10E6nOXhRERhAmRVWla5i99jeUGWmI-Xl?usp=sharing)

Download and extract NVD dataset and keep it in the ```Dataset/NVD/Processed``` directory. Or execute ```PrepareDataset.ipynb``` or ```PrepareDataset.py``` to download and prepare dataset.

### Create or manage virtual environment if Anaconda is available in the system
Check your system if Anaconda module is available. If anaconda is not available install packages in the python base. If anaconda is available, then create a virtual enviroment to manage python packages.  

1. Load Module: ```load module anaconda/version_xxx```
2. Create virtual environment: ```conda create -n v2wbert python=3.7```. Here python version 3.7 is considered.
3. Activate virtual environement: ```conda activate v2wbert``` or ```source activate v2wbert```

Other necessary commands for managing enviroment can be found here : [https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

### Installation of pacakages
The installations are considered for python version 3.7

Most python packages are intalled using ```pip``` or ```conda``` command. For consistency it's better to follow only one of them. If anaconda not available install packages in python base using ```pip``` command.

#### Pytorch
Link of Pytorch installation is here: [https://pytorch.org/](https://pytorch.org/).
If Pytorch is already installed then this is not necessary.

#### Installation of Tensorflow
Only some functionalities of tensorflow is used in the project. If tensorflow is not available in the system, I will try to replace those with another function. Any version of tensorflow will do.

[https://www.tensorflow.org/overview/](https://www.tensorflow.org/overview/)


#### Numpy

Command:  ```pip install numpy```

More details can be found here, [https://numpy.org/install/](https://numpy.org/install/)


#### Install Pandas

Command: ```pip install pandas```

[https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html](https://pandas.pydata.org/pandas-docs/stable/getting_started/install.html)


#### Installation of Transformers for BERT and other libraries

We will be using HuggingFace ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)) library for transformers.

```
pip install transformers
pip install wget
pip install ipywidgets
```


#### Package `ipynb ` for calling one python functions from another Jupyter notebook file

```
pip install ipynb
```

#### install `beutifulsoup` for html xml parsing
This is not necessary now but later.

```
pip install beautifulsoup4
pip install lxml
```


## Running

#### Prepare dataset 

- Run ```PrepareDataset.ipynb``` notebook for download and prepare dataset

```
python PrepareDataset.py --dir='Dataset' --from_year=2020 --to_year=2021 --from_train_year=1990 --to_train_year=2020 --from_test_year=2021 --to_test_year=2021 --from_val_year=2022 --to_test_year=2022
```
####  Pretraining dataset 

-- Pretraining ```V2WBERT-Pretraining.ipynb```

```
python V2WBERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200
```
---

###### Other options 
- Running 'dummy' dataset to test the overall process
```
python V2W-BERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='dummy'
```

- Temporal dataset splits data by year
```
python V2W-BERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='temporal'
```

- Random dataset splits data from each category
```
python V2W-BERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='dp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='random'
```

- To run in distributed dataparallel mode

```
python V2W-BERT-Pretraining.py --pretrained='distilbert-base-uncased' --num_gpus=2 --parallel_mode='ddp' --epochs=30 --batch_size=16 --refresh_rate=200 --rand_dataset='random'
```

#### Link Prediction

- Link Prediciton ```V2WBERT-LinkPrediction.ipynb```

```
python V2W-BERT-LinkPrediction.py --pretrained='distilbert-base-uncased' --use_pretrained=True --use_rd=False --checkpointing=False --rand_dataset='temporal'  --performance_mode=False --neg_link=128  --epoch=25 --nodes=1 --num_gpus=2 --batch_size=64
```

###### Other options

- other dataset
```
python V2W-BERT-LinkPrediction.py --pretrained='distilbert-base-uncased' --use_pretrained=True --use_rd=False --checkpointing=False --rand_dataset='random'  --performance_mode=False --neg_link=128  --epoch=25 --nodes=1 --num_gpus=2 --batch_size=64
```
- Running 'dummy' dataset to test the overall process
```
python V2W-BERT-LinkPrediction.py --pretrained='distilbert-base-uncased' --use_pretrained=True --use_rd=False --checkpointing=False --rand_dataset='dummy'  --performance_mode=False --neg_link=128  --epoch=25 --nodes=1 --num_gpus=2 --batch_size=64
```




## Cite

Please cite [our paper](https://ieeexplore.ieee.org/document/9564227) if you use this code in your own work:

```
@inproceedings{das2021v2w,
title={V2W-BERT: A Framework for Effective Hierarchical Multiclass Classification of Software Vulnerabilities},
  author={Das, Siddhartha Shankar and Serra, Edoardo and Halappanavar, Mahantesh and Pothen, Alex and Al-Shaer, Ehab},
  booktitle={2021 IEEE 8th International Conference on Data Science and Advanced Analytics (DSAA)},
  pages={1--12},
  year={2021},
  organization={IEEE}
}
```

Feel free to [email us](mailto:das90@purdue.edu) for additional resources.
If you notice anything unexpected, please open an [issue](https://github.com/Cybersecurity-Tools/V2W-BERT) and let us know.
If you have any questions or are missing a specific feature, feel free [to discuss them with us](https://github.com/Cybersecurity-Tools/V2W-BERT/discussions/).


## Copyright
This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.
Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.

<p align="center">
    PACIFIC NORTHWEST NATIONAL LABORATORY</br>
	<i>operated by</i></br>
	BATTELLE</br>
	<i>for the</i></br>
	UNITED STATES DEPARTMENT OF ENERGY</br>
	<i>under Contract DE-AC05-76RL01830<i/>
</p>

