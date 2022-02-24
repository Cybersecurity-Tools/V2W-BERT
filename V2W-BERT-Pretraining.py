#!/usr/bin/env python
# coding: utf-8

# ***********************************************************************
# 
#         V2W-BERT: A Python library for vulnerability classification
#                Siddhartha Das (das90@purdue.edu) : Purdue University
#                Mahantesh Halappanavar (hala@pnnl.gov): Pacific Northwest National Laboratory   
# 
# ***********************************************************************
# 
#  
# Copyright © 2022, Battelle Memorial Institute
# All rights reserved.
# 
#  
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#  
# 1. Redistributions of source code must retain the above copyright notice, this
# 
#    list of conditions and the following disclaimer.
# 
#  
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
# 
#    this list of conditions and the following disclaimer in the documentation
# 
#    and/or other materials provided with the distribution.
# 
#  
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# 
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# 
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# 
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# 
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# 
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# 
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# 
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# 
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# ### Packages

# In[70]:


import torch# If there's a GPU available...
import random
import numpy as np
import multiprocessing
import pandas as pd
import time

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" ##I will find a way to fix this later :(

NUM_GPUS=0

try:
    if torch.cuda.is_available():  
        device = torch.device("cuda")
        NUM_GPUS=torch.cuda.device_count()
        print('There are %d GPU(s) available.' % NUM_GPUS)
        print('We will use the GPU:', torch.cuda.get_device_name())# If not...
    else:
        print('No GPU available, using the CPU instead.')
        device = torch.device("cpu")  
except:
    print('Cuda error using CPU instead.')
    device = torch.device("cpu")  
    
print(device)

# device = torch.device("cpu")  
# print(device)

NUM_PROCESSORS=multiprocessing.cpu_count()
print("Cpu count: ",NUM_PROCESSORS)


# #### Specify Directories

# In[71]:


from ipynb.fs.full.Dataset import getDataset, getDummyDataset, Data        

DIR='./Results'
    
from pathlib import Path
Path(DIR).mkdir(parents=True, exist_ok=True)

DATASET_LOAD_DIR="./Dataset/NVD/processed/"
MODEL_SAVE_DIR=DIR+'/Model/'

Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)
print("Data loading directory: ", DIR)
print("Model Saving directory:", MODEL_SAVE_DIR)


# ### Some more packages

# In[72]:


import pandas as pd
import pathlib
import zipfile
import wget

import torch
from torch import nn
from torch import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import random_split
from torch.utils.data import RandomSampler,SequentialSampler
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from transformers import AutoModelForMaskedLM
from transformers import AutoConfig

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import DataCollatorForLanguageModeling

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.tuner.tuning import Tuner
from pytorch_lightning.callbacks import ModelCheckpoint


# #### For reproduciblity

# In[73]:


# Set the seed value all over the place to make this reproducible.
from random import sample

seed_val = 42
os.environ['PYTHONHASHSEED'] = str(seed_val)
random.seed(seed_val)
np.random.seed(seed_val)
torch.manual_seed(seed_val)
pl.seed_everything(seed_val)

try:
    torch.cuda.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
except:
    print("nothing to set for cudnn")


# ### Model definition

# Possible choices for pretrained are:
# distilbert-base-uncased
# bert-base-uncased
# 
# The BERT paper says: "[The] pre-trained BERT model can be fine-tuned with just one additional output
# layer to create state-of-the-art models for a wide range of tasks, such as question answering and
# language inference, without substantial task-specific architecture modifications."
# 
# Huggingface/transformers provides access to such pretrained model versions, some of which have been
# published by various community members.
# 
# BertForSequenceClassification is one of those pretrained models, which is loaded automatically by
# AutoModelForSequenceClassification because it corresponds to the pretrained weights of
# "bert-base-uncased".
# 
# Huggingface says about BertForSequenceClassification: Bert Model transformer with a sequence
# classification/regression head on top (a linear layer on top of the pooled output) e.g. for GLUE
# tasks."
# 
# This part is easy  we instantiate the pretrained model (checkpoint)
# 
# But it's also incredibly important, e.g. by using "bert-base-uncased, we determine, that that model
# does not distinguish between lower and upper case. This might have a significant impact on model
# performance!!!

# In[74]:


class Model(pl.LightningModule):
    def __init__(self,*args, **kwargs):
        super().__init__()

        self.save_hyperparameters()
        # a very useful feature of pytorch lightning  which leads to the named variables that are passed in
        # being available as self.hparams.<variable_name> We use this when refering to eg
        # self.hparams.learning_rate

        # freeze
        self._frozen = False

        # eg https://github.com/stefan-it/turkish-bert/issues/5
        config = AutoConfig.from_pretrained(self.hparams.pretrained,
                                            output_attentions=False,
                                            output_hidden_states=False)

        #print(config)

        A = AutoModelForMaskedLM
        self.model = A.from_pretrained(self.hparams.pretrained, config=config)

        print('Model: ', type(self.model))
        

    def forward(self, batch):
        # there are some choices, as to how you can define the input to the forward function I prefer it this
        # way, where the batch contains the input_ids, the input_put_mask and sometimes the labels (for
        # training)
        
        #print(batch['input_ids'].shape)
        #print(batch['labels'].shape)
                
        outputs = self.model(input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'])

        loss = outputs[0]

        return loss
    
#     def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
#         self.t2=time.time()
    
#     def on_train_epoch_start(self):
#         self.t0=time.time()
    
    
#     def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        
#         if batch_idx%50 ==0:
#             t1=time.time()        
#             #print("Batch {0} of {1}: {2:.6f}".format(batch_idx+1, self.total_train_batches, t1-self.t0))
#             print("Batch {0}: {1:.4f}".format(batch_idx, t1-self.t0))
    
        
    def training_step(self, batch, batch_idx):
        # the training step is a (virtual) method,specified in the interface, that the pl.LightningModule
        # class stipulates you to overwrite. This we do here, by virtue of this definition
        
        # self refers to the model, which in turn acceses the forward method
        loss = self(batch)
        
        #tensorboard_logs = {'train_loss': loss}
        # pytorch lightning allows you to use various logging facilities, eg tensorboard with tensorboard we
        # can track and easily visualise the progress of training. In this case
        
        self.log('loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        #self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        #return {'loss': loss, 'log': tensorboard_logs}
        # the training_step method expects a dictionary, which should at least contain the loss
        return loss
    
    def validation_step(self, batch, batch_idx):
        val_loss = self(batch)
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)

        return val_loss
        
    def test_step(self, batch, batch_idx):
        test_loss = self(batch)
        self.log('test_loss', test_loss, on_epoch=True, prog_bar=True)
        
        return test_loss
    
#     def on_train_epoch_end(self, train_step_outputs):
#         print('Epoch ',self.current_epoch,' done: took ',time.time()-self.t0, ' sec')

        #---------------------        
#         print(train_step_outputs)        
#         import pdb; pdb.set_trace()

#         avg_loss = torch.stack([x['loss'] for x in train_step_outputs]).mean()
    
#         print(torch.stack([x['loss'] for x in train_step_outputs]))
#         tensorboard_logs = {'train_loss': avg_loss}
        
#         self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
#         self.log('log', tensorboard_logs, on_step=True, on_epoch=True, prog_bar=True)
        
#         return {
#             'train_loss': avg_loss,
#             'log': tensorboard_logs,
#             'progress_bar': {
#                 'train_loss': avg_loss
#             }
#         }
#---------------------

        
    def configure_optimizers(self):
        # The configure_optimizers is a (virtual) method, specified in the interface, that the
        # pl.LightningModule class wants you to overwrite.
        # In this case we define that some parameters are optimized in a different way than others. In
        # particular we single out parameters that have 'bias', 'LayerNorm.weight' in their names. For those
        # we do not use an optimization technique called weight decay.

        no_decay = ['bias', 'LayerNorm.weight']

        optimizer_grouped_parameters = [{
            'params': [
                p for n, p in self.named_parameters()
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.01
        }, {
            'params': [
                p for n, p in self.named_parameters()
                if any(nd in n for nd in no_decay)
            ],
            'weight_decay':
            0.0
        }]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.learning_rate,
                          eps=1e-8 # args.adam_epsilon  - default is 1e-8.
                          )

        
        # We also use a scheduler that is supplied by transformers.
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0, # Default value in run_glue.py
            num_training_steps=self.hparams.num_training_steps)

        return [optimizer], [scheduler]

#---------------------
#     def freeze(self) -> None:
#         # freeze all layers, except the final classifier layers
#         for name, param in self.model.named_parameters():
#             if 'classifier' not in name:  # classifier layer
#                 param.requires_grad = False


#         self._frozen = True

#     def unfreeze(self) -> None:
#         if self._frozen:
#             for name, param in self.model.named_parameters():
#                 if 'classifier' not in name:  # classifier layer
#                     param.requires_grad = True

#         self._frozen = False

#     def on_epoch_start(self):
#         """pytorch lightning hook"""
#         if self.current_epoch < self.hparams.nr_frozen_epochs:
#             self.freeze()

#         if self.current_epoch >= self.hparams.nr_frozen_epochs:
#             self.unfreeze()
#---------------------


# ### Data class definition

# So here we finally arrive at the definition of our data class derived from pl.LightningDataModule.
# 
# In earlier versions of pytorch lightning  (prior to 0.9) the methods here were part of the model class
# derived from pl.LightningModule. For better flexibility and readability the Data and Model related parts
# were split out into two different classes:
# 
# pl.LightningDataModule and pl.LightningModule
# 
# with the Model related part remaining in pl.LightningModule
# 
# This is explained in more detail in this video: https://www.youtube.com/watch?v=L---MBeSXFw
# ```
# class CDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, labels):
#         self.encodings = encodings
#         self.labels = labels
# 
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
#         item['labels'] = torch.tensor(self.labels[idx])
#         
#         return item
# 
#     def __len__(self):
#         return len(self.labels)
# ```
# 
# 
# Another testing code
# 
# ```
# class CDataset(torch.utils.data.Dataset):
#     def __init__(self, encodings, collator, labels):
#         self.encodings = encodings
#         self.collator = collator
#         
#     def __getitem__(self, idx):
#         item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}        
#         item = self.collator([item])
#         item = {key: val[0] for key, val in item.items()}
#     
#         return item
# 
#     def __len__(self):
#         return len(self.labels)
# 
# A=AutoTokenizer
# berttokenizer=A.from_pretrained('bert-base-uncased')
# datacollator=DataCollatorForLanguageModeling(tokenizer=berttokenizer,mlm_probability=0.15, mlm=True)
# 
# data, sentences, labels = getDummyDataset()
# 
# train_encodings = berttokenizer(sentences, truncation=True, padding=True)
# 
# dataset = CDataset(train_encodings,datacollator)
# 
# dataset[0]
# ```
# 
# cf this open issue: https://github.com/PyTorchLightning/pytorch-lightning/issues/3232

# In[75]:


class CDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, collator):
        self.encodings = encodings
        self.labels = labels
        self.collator = collator

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}            
        item = self.collator([item])
        item = {key: val[0] for key, val in item.items()}
    
        if type(self.labels)!=torch.Tensor:
            item['org_labels']= torch.tensor(self.labels[idx])
        else:
            item['org_labels']= self.labels[idx]
    
        return item

    def __len__(self):
        return len(self.labels)


class DataProcessing(pl.LightningDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

#         self.save_hyperparameters()
        if isinstance(args, tuple): args = args[0]
        self.hparams = args
        self.batch_size=self.hparams.batch_size        

#         print('args:', args)
#         print('kwargs:', kwargs)
#         print(f'self.hparams.pretrained:{self.hparams.pretrained}')

        #print('Loading BERT tokenizer')
        print(f'PRETRAINED:{self.hparams.pretrained}')

        A = AutoTokenizer
        self.tokenizer = A.from_pretrained(self.hparams.pretrained, use_fast=True)

        print('Tokenizer:', type(self.tokenizer))
        
        self.datacollator=DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        
    def setup(self, stage=None):
        
        MAX_TEXT_LENGTH=512
    
#         CVE dataset
#         ------------------------------------        
        data, df_CVE, df_CWE=None,None,None
    
        if self.hparams.rand_dataset=='dummy':            
            #------------------------------------
            data, sentences, labels = getDummyDataset()        
            #------------------------------------
        else:        
            if self.hparams.rand_dataset=='temporal':
                print("Temporal Partition:--")
                data, df_CVE, df_CWE = getDataset(DATASET_LOAD_DIR)            
            else:
                print("Random Partition:--")
                data, df_CVE, df_CWE = getRandomDataset(DATASET_LOAD_DIR, 0.70, 0.10)
                
            #print(df_CVE)

            sentences=df_CVE['CVE Description'].apply(lambda x: str(x)[:MAX_TEXT_LENGTH])
            
            #print(sentences)
            
            labels=data.y
            CWE_IDS_USED=df_CWE['Name'].tolist()
            INDEX_TO_CWE_MAP=dict(zip(list(range(len(CWE_IDS_USED))),CWE_IDS_USED))
            CWE_TO_INDEX_MAP=dict(zip(CWE_IDS_USED,list(range(len(CWE_IDS_USED)))))
            sentences=sentences.tolist()
        
        
        if type(labels)!=torch.Tensor:
            labels=torch.tensor(labels,dtype=torch.long)
        else:
            labels=labels.type(torch.LongTensor)
        
        self.NUM_CLASSES=len(data.y[0])
    
        train_encodings = self.tokenizer(sentences, truncation=True, padding=True, max_length=MAX_TEXT_LENGTH)        
        self.dataset = CDataset(train_encodings, labels, self.datacollator)        
        
        val_mask= (data.val_mask == True).nonzero().flatten().numpy()
        val_encodings = self.tokenizer([sentences[i] for i in val_mask], truncation=True, padding=True, max_length=MAX_TEXT_LENGTH)
        self.val_dataset=CDataset(val_encodings, labels[data.val_mask], self.datacollator)
        
        test_mask= (data.test_mask == True).nonzero().flatten().numpy()
        test_encodings = self.tokenizer([sentences[i] for i in test_mask], truncation=True, padding=True, max_length=MAX_TEXT_LENGTH)
        self.test_dataset=CDataset(test_encodings, labels[data.test_mask], self.datacollator)
                
        #print('Example Sentence[0]: ', sentences[0])              
    
    
    def train_dataloader(self):
        
        train_sampler = RandomSampler(self.dataset)
        
        return DataLoader(self.dataset,
                         #sampler=train_sampler, 
                          batch_size=self.batch_size,
                          num_workers=min(NUM_PROCESSORS,self.batch_size)
                         )
    
    def val_dataloader(self):
        
        val_sampler = SequentialSampler(self.val_dataset)
        
        return DataLoader(self.val_dataset,
                          sampler=val_sampler, 
                          batch_size=self.batch_size,
                          num_workers=min(NUM_PROCESSORS,self.batch_size)
                         )
    
    def test_dataloader(self):
        
        test_sampler = SequentialSampler(self.test_dataset)
        
        return DataLoader(self.test_dataset,
                          sampler=test_sampler, 
                          batch_size=self.batch_size,
                          num_workers=min(NUM_PROCESSORS,self.batch_size)
                         )


# In[76]:


def printModelParams(model):
    print (model)
    # Get all of the model's parameters as a list of tuples.
    params = list(model.named_parameters())
    print('The model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')
    for p in params[0:5]:
        print("{:<55} {:>12}, {}".format(p[0], str(tuple(p[1].size())),p[1].requires_grad))

    print('\n==== First Transformer ====\n')
    for p in params[5:21]:
        print("{:<55} {:>12}, {}".format(p[0], str(tuple(p[1].size())),p[1].requires_grad))

    print('\n==== Output Layer ====\n')
    for p in params[-5:]:
        print("{:<55} {:>12}, {}".format(p[0], str(tuple(p[1].size())),p[1].requires_grad))


# In[77]:


def print_model_value(model):
    params = list(model.named_parameters())
    print (params[-1][0],params[-1][1][:4])


# ### Main
# Two key aspects:
# 
# - pytorch lightning can add arguments to the parser automatically
# - you can manually add your own specific arguments.
# 
# - there is a little more code than seems necessary, because of a particular argument the scheduler
#   needs. There is currently an open issue on this complication
#   https://github.com/PyTorchLightning/pytorch-lightning/issues/1038

# ### Automatic Batching (not used)

# ```
# if args.auto_batch>0:    
#         #init_batch_size=32
#         init_batch_size=args.auto_batch
#         tuner = Tuner(trainer)        
#         assert hasattr(dataProcessor, "batch_size")
#         new_batch_size = tuner.scale_batch_size(model, 
#                                                 mode="binsearch", 
#                                                 init_val=init_batch_size, 
#                                                 max_trials=10,
#                                                 datamodule=dataProcessor,                                        
#                                                )
# 
#         print("Max batch size: ", new_batch_size)
#         #dataProcessor.batch_size = new_batch_size
# ```

# ### Get Configuration to run

# In[78]:


import argparse
from argparse import ArgumentParser

def get_configuration():
    parser = ArgumentParser()

    #parser.add_argument('--pretrained', type=str, default="bert-base-uncased")
    #parser.add_argument('--pretrained', type=str, default="roberta-base")
    parser.add_argument('--pretrained', type=str, default="distilbert-base-uncased") 
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--nr_frozen_epochs', type=int, default=5)
    parser.add_argument('--training_portion', type=float, default=0.9)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--auto_batch', type=int, default=-1)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--frac', type=float, default=1)
    parser.add_argument('--num_gpus', type=int, default=-1)
    parser.add_argument('--nodes', type=int, default=1)
    parser.add_argument('--parallel_mode', type=str, default="dp", choices=['dp', 'ddp', 'ddp2'])
    parser.add_argument('--refresh_rate', type=int, default=1)
    parser.add_argument('--check', type=bool, default=False)
    parser.add_argument('--rand_dataset', type=str, default="temporal", choices=['temporal','random','dummy'])
    
    
    parser.add_argument('-f') ##dummy for jupyternotebook

    # parser = Model.add_model_specific_args(parser) parser = Data.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    
    print("-"*50)
    print("BATCH SIZE: ", args.batch_size)
    
    # start : get training steps
    dataProcessor = DataProcessing(args)
    dataProcessor.setup()
    
    args.num_training_steps = len(dataProcessor.train_dataloader())*args.epochs
    dict_args = vars(args)
    
    gpus=-1
    if NUM_GPUS>0:
        gpus=args.num_gpus        
    else:
        args.parallel_mode=None
        gpus=None
    
    print("USING GPUS:", gpus)
    print("-"*50)
    
    # saves a file like: my/path/sample-mnist-epoch=02-val_loss=0.32.ckpt
    checkpoint_callback = ModelCheckpoint(
        monitor='loss_epoch',
        dirpath=MODEL_SAVE_DIR,
        #filename='{epoch:02d}-{loss:.4f}',
        filename="V2W-BERT"+args.pretrained+'-{epoch:02d}-{loss:.4f}',
        save_top_k=1,
        mode='min',
        save_weights_only=True,
        #prefix="CBERT-"+args.pretrained,#+'-'+str(args.parallel_mode),
        save_last=True,
    )
    
#     if args.check==False:
#         args.checkpoint_callback = False
#     elif args.parallel_mode=='dp':
#         args.callbacks=[checkpoint_callback]        
#     else:
#         args.checkpoint_callback = False

    args.checkpoint_callback = False
    
    trainer = pl.Trainer.from_argparse_args(args, 
                                            gpus=gpus,
                                            num_nodes=args.nodes, 
                                            accelerator=args.parallel_mode,
                                            max_epochs=args.epochs, 
                                            gradient_clip_val=1.0,                                            
                                            logger=False,
                                            progress_bar_refresh_rate=args.refresh_rate,
                                            profiler='simple', #'simple',
                                            default_root_dir=MODEL_SAVE_DIR,                                            
                                            deterministic=True,
                                           )

    return trainer, dataProcessor, args, dict_args

# trainer, dataProcessor, args, dict_args = get_configuration()
# next(iter(dataProcessor.test_dataloader()))


# In[79]:


def train_model():    
    trainer, dataProcessor, args, dict_args = get_configuration()
    
    model = Model(**dict_args)    
    
#     printModelParams(model)
#     args.early_stop_callback = EarlyStopping('val_loss')
    
    
    print("Original weights: ");print_model_value(model)
    
    t0=time.time()
    trainer.fit(model, dataProcessor)
    print('Training took: ',time.time()-t0)
    
    print("Trained weights: ");print_model_value(model)
    
    #if args.parallel_mode!='dp':    
    print("Saving the last model")
    #MODEL_NAME=MODEL_SAVE_DIR+"CBERT-"+args.pretrained+'-'+args.parallel_mode+".ckpt"
    MODEL_NAME=MODEL_SAVE_DIR+"V2WBERT-"+args.pretrained+".ckpt"
    trainer.save_checkpoint(MODEL_NAME)

    print("Testing:....")
    trainer.test(model, dataProcessor.test_dataloader())
    
    print("Training Phase Complete......")


# In[80]:


def test_model():
    trainer, dataProcessor, args, dict_args = get_configuration()
    
    #MODEL_NAME=MODEL_SAVE_DIR+"CBERT-"+args.pretrained+'-'+args.parallel_mode+".ckpt"    
    MODEL_NAME=MODEL_SAVE_DIR+"V2WBERT-"+args.pretrained+".ckpt"    
#     if args.parallel_mode=='dp':
#         MODEL_NAME=MODEL_SAVE_DIR+"CBERT-"+args.pretrained+'-'+args.parallel_mode+"-last.ckpt"
    
    if os.path.exists(MODEL_NAME): 
        print('Loading Saved Model: ',MODEL_NAME)        
    else: 
        print("File not found: ",MODEL_NAME)
        return
    
    model=None
    
    if args.parallel_mode!='dp':
        model = Model.load_from_checkpoint(MODEL_NAME)
    else:
        model = Model(**dict_args)
        print("Original weights: ");print_model_value(model)
        checkpoint = torch.load(MODEL_NAME, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])

    print("Loaded weights: ");print_model_value(model)    
    trainer.test(model, dataProcessor.test_dataloader())    
    print("Test Complete......")
    
    return model


# In[81]:


if __name__ == "__main__":
    train_model()
    test_model()
    


# In[ ]:




