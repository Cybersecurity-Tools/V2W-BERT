#!/usr/bin/env python
# coding: utf-8
# // ***********************************************************************
# //
# //        V2W-BERT: A Python library for vulnerability classification
# //               Siddhartha Das (das90@purdue.edu) : Purdue University
# //               Mahantesh Halappanavar (hala@pnnl.gov): Pacific Northwest National Laboratory   
# //
# // ***********************************************************************

 
# Copyright © 2022, Battelle Memorial Institute
# All rights reserved.

 

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

 
# 1. Redistributions of source code must retain the above copyright notice, this

#    list of conditions and the following disclaimer.

 

# 2. Redistributions in binary form must reproduce the above copyright notice,

#    this list of conditions and the following disclaimer in the documentation

#    and/or other materials provided with the distribution.

 

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"

# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE

# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE

# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE

# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL

# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR

# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER

# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,

# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE

# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# ## Download and Preprocess Latest Dataset
# 
# In this script we first download all CVEs to-date. Use the NVD and Mitre hierarchy documents to prepare a train test validation set.

# ## Import libraries

# In[199]:


import os
import requests, zipfile, io
import pickle
import pandas as pd
import numpy as np
# Here, I have disabled a false alarm that would otherwise trip later in the project.
pd.options.mode.chained_assignment = None

# The datetime library will let me filter the data by reporting date.
from datetime import datetime, timedelta
# Since the NVD data is housed in JavaScript Object Notation (JSON) format, I will need the json_normalize function to access and manipulate the information.
from pandas.io.json import json_normalize
import sys
import torch
import re

from ipynb.fs.full.Dataset import  Data        


# In[200]:


# Expanding view area to facilitate data manipulation.
pd.set_option('display.max_rows', 20)
pd.set_option('display.max_columns', 100)


# In[201]:


import argparse
from argparse import ArgumentParser

def get_configuration():
    parser = ArgumentParser()

    parser.add_argument('--dir', type=str, default='Dataset')
    
    parser.add_argument('--from_year', type=int, default=2020)
    parser.add_argument('--to_year', type=int, default=2022)
    
    parser.add_argument('--from_train_year', type=int, default=1990)
    parser.add_argument('--to_train_year', type=int, default=2020)
    
    parser.add_argument('--from_test_year', type=int, default=2021)
    parser.add_argument('--to_test_year', type=int, default=2021)
    
    parser.add_argument('--from_val_year', type=int, default=2022)
    parser.add_argument('--to_val_year', type=int, default=2022)
    
    
    parser.add_argument('-f') ##dummy for jupyternotebook

    args = parser.parse_args()    
    dict_args = vars(args)

    return args, dict_args

args, dict_args=get_configuration()

print(dict_args)
print(args.dir)


# In[ ]:





# ### Configuration

# In[202]:


class DataPath():
    def __init__(self, args, dataset_dir='',results_dir=''):
        
        #File locations
        self.PATH_TO_DATASETS_DIRECTORY = dataset_dir+'/NVD/raw/'
        self.PATH_TO_RESULTS_DIRECTORY = results_dir+'/NVD/processed/'
        
        self.NVD_CVE_FILE=self.PATH_TO_RESULTS_DIRECTORY+'NVD_CVE_data.csv'
        self.Graph_FILE=self.PATH_TO_RESULTS_DIRECTORY+'GRAPH_data'
        self.GRAPHVIZ_HIERARCHY=self.PATH_TO_RESULTS_DIRECTORY+'Hierarchy'
        
        self.MITRE_CWE_FILE=self.PATH_TO_DATASETS_DIRECTORY+'CWE_RC_1000.csv'
        self.NVD_CWE_FILE=self.PATH_TO_RESULTS_DIRECTORY+'NVD_CWE_data.csv'        
        
        self.MASK_FILE = self.PATH_TO_RESULTS_DIRECTORY+'NVD_data'
        self.MERGED_NVD_CVE_FILE=self.PATH_TO_RESULTS_DIRECTORY+'NVD_CVE.csv'
        self.FILTERED_NVD_CWE_FILE=self.PATH_TO_RESULTS_DIRECTORY+'NVD_CWE.csv'
        
        self.YEARS=list(range(args.from_year,args.to_year+1))
        
        self.TRAIN_YEARS=list(range(args.from_train_year,args.to_train_year+1))
        self.VAL_YEARS=list(range(args.from_val_year,args.to_val_year+1))
        self.TEST_YEARS=list(range(args.from_test_year,args.to_test_year+1)) 
        
        
        if not os.path.exists(self.PATH_TO_DATASETS_DIRECTORY):
            print("Creating directory: ",self.PATH_TO_DATASETS_DIRECTORY)
            os.makedirs(self.PATH_TO_DATASETS_DIRECTORY)
        if not os.path.exists(self.PATH_TO_RESULTS_DIRECTORY):
            print("Creating directory: ",self.PATH_TO_RESULTS_DIRECTORY)
            os.makedirs(self.PATH_TO_RESULTS_DIRECTORY)


class Config(DataPath):
    def __init__(self,args, dataset_dir='',results_dir=''):
        super(Config, self).__init__(args, dataset_dir, results_dir)
        self.CLUSTER_LABEL=0
    
        self.download()
        
    def download(self):        
        for year in self.YEARS:            
            if not os.path.exists(self.PATH_TO_DATASETS_DIRECTORY+'nvdcve-1.1-'+str(year)+'.json'):
                url = 'https://nvd.nist.gov/feeds/json/cve/1.1/nvdcve-1.1-'+str(year)+'.json.zip'
                print("Downloading: ",url)
                r = requests.get(url)
                z = zipfile.ZipFile(io.BytesIO(r.content))
                z.extractall(self.PATH_TO_DATASETS_DIRECTORY)
                print("CVEs downloaded")
         
        if not os.path.exists(self.MITRE_CWE_FILE):
            url = 'https://drive.google.com/uc?export=download&id=1-phSamb4RbxyoBc3AQ2xxKMSsK2DwPyn'
            print("Downloading: ",url)
            r = requests.get(url)
            z = zipfile.ZipFile(io.BytesIO(r.content))
            z.extractall(self.PATH_TO_DATASETS_DIRECTORY)
            print("CWEs downloaded")
            

config=Config(args,dataset_dir=args.dir,results_dir=args.dir)


# ### ProfecessCVES

# In[203]:


def getDataFrame(config):
    df = []
    counter=0
    for year in config.YEARS:    
        yearly_data = pd.read_json(config.PATH_TO_DATASETS_DIRECTORY+'nvdcve-1.1-'+str(year)+'.json')        
        
        if counter == 0:
            df = yearly_data
        else:
            df = df.append(yearly_data)
        counter+=1
        
    return df


# In[204]:


def removeREJECT(description):
    series=[]
    for x in description:
        try:        
            if "REJECT" in (json_normalize(x)["value"])[0]:                    
                series.append(False)
            else:
                series.append(True)            
        except:
            series.append(False)
    
    return pd.Series(series,index=description.index)


# In[205]:


def removeUnknownCWE(description):    
    series=[]
    for x in description:
        try:        
            if x == "UNKNOWN" or x == "NONE":
                series.append(False)
            else:
                series.append(True)            
        except:
            series.append(False)
    
    return pd.Series(series,index=description.index)


# In[206]:


def getCVEDescription(df):
    CVE_entry = []
    CVE_index = df["cve.description.description_data"].index

    for x in df["cve.description.description_data"]:
        try:
            raw_CVE_entry = json_normalize(x)["value"][0]
            clean_CVE_entry = str(raw_CVE_entry)
            CVE_entry.append(clean_CVE_entry)
        except:
            CVE_entry.append("NONE")
    CVE_entry = pd.Series(CVE_entry, index = CVE_index)
    
    return  CVE_entry


# In[207]:


# Defining a function which I will use below
def consolidate_unknowns(x):
    if x == "NVD-CWE-Other" or x == "NVD-CWE-noinfo":
        return "UNKNOWN"
    else:
        return x


# In[208]:


def getCWEs(df):
    CWE_entry = []
    CWE_index = df["cve.problemtype.problemtype_data"].index

    for x in df["cve.problemtype.problemtype_data"]:
        try:
            CWE_normalized_json_step_1 = json_normalize(x)
            CWE_normalized_json_step_2 = CWE_normalized_json_step_1["description"][0]
            CWEs=[]
            #print(json_normalize(CWE_normalized_json_step_2)["value"])
            for CWE in json_normalize(CWE_normalized_json_step_2)["value"]:
                #CWEs.append(consolidate_unknowns(str(CWE)))
                CWEs.append(str(CWE))
            CWE_entry.append(CWEs)
        except:
            CWE_entry.append(['NONE'])
    CWE_entry = pd.Series(CWE_entry, index = CWE_index)
    
    return  CWE_entry


# In[209]:


def ProcessDataset(config):
    print("Loading data from file---")
    df=getDataFrame(config)
    CVE_Items = json_normalize(df["CVE_Items"])
    df = pd.concat([df.reset_index(), CVE_Items], axis=1)
    df = df.drop(["index", "CVE_Items"], axis=1)
    
    df = df.rename(columns={"cve.CVE_data_meta.ID": "CVE ID"})
    CVE_ID = df["CVE ID"]
    df.drop(labels=["CVE ID"], axis=1,inplace = True)
    df.insert(0, "CVE ID", CVE_ID)
    
    ##remove description with REJECT
    print("Removing REJECTs---")
    df=df[removeREJECT(df["cve.description.description_data"])]
    
    ##Extract CVE description
    CVE_description=getCVEDescription(df)
    df.insert(1, "CVE Description", CVE_description)
        
    ##Extract CWEs
    print("Extracting CWEs---")
    CWE_entry=getCWEs(df)
    df.insert(2, "CWE Code", CWE_entry)
    
#     ##Remove CWEs we don't know true label
#     print("Removing Unknown CWEs---")
#     df=df[removeUnknownCWE(df["CWE Code 1"])]
    
    # Converting the data to pandas date-time format
    df["publishedDate"] = pd.to_datetime(df["publishedDate"])
        
    return df


# ### ProcessCWEs

# In[210]:


def processAndSaveCVE(config, LOAD_SAVED=True):
    
    if not os.path.exists(config.NVD_CVE_FILE) or LOAD_SAVED==False:
        df=ProcessDataset(config)
        df=df[['publishedDate', 'CVE ID', 'CVE Description', 'CWE Code']]                
        df.to_csv(config.NVD_CVE_FILE,index=False)
    else:
        df=pd.read_csv(config.NVD_CVE_FILE)
            
    return df


# In[211]:


def ProcessCWE_NVD(config):
    # Importing BeautifulSoup and an xml parser to scrape the CWE definitions from the NVD web site
    from bs4 import BeautifulSoup
    import lxml.etree
    
    # loading the NVD CWE Definitions page and scraping it for the first table that appears
    NVD_CWE_description_url = requests.get("https://nvd.nist.gov/vuln/categories")
    CWE_definitions_page_soup =  BeautifulSoup(NVD_CWE_description_url.content, "html.parser")
    table = CWE_definitions_page_soup.find_all('table')[0] 
    df_CWE_definitions = pd.read_html(str(table))[0]
    
    return df_CWE_definitions
    


# In[212]:


def ProcessCWE_MITRE(config):
    print('Loading CWE file : {0}'.format(config.MITRE_CWE_FILE))
    #df_CWE_definitions = pd.read_csv(config.MITRE_CWE_FILE, quotechar='"',delimiter=',', encoding='latin1',index_col=False)    
    df_CWE_definitions = pd.read_csv(config.MITRE_CWE_FILE, delimiter=',', encoding='latin1',index_col=False)
    
    return df_CWE_definitions


# In[213]:


def processAndSaveCWE(config, LOAD_SAVED=True):
    
    if not os.path.exists(config.MITRE_CWE_FILE) or LOAD_SAVED==False:        
        df_CWE_MITRE=ProcessCWE_MITRE(config)
        df_CWE_MITRE.to_csv(config.MITRE_CWE_FILE,index=False)
    else:
        df_CWE_MITRE=pd.read_csv(config.MITRE_CWE_FILE, index_col=False)
    
    
    if not os.path.exists(config.NVD_CWE_FILE) or LOAD_SAVED==False:        
        df_CWE_NVD=ProcessCWE_NVD(config)
        df_CWE_NVD.to_csv(config.NVD_CWE_FILE,index=False)
    else:
        df_CWE_NVD=pd.read_csv(config.NVD_CWE_FILE,index_col=False)
    
    
    return df_CWE_MITRE, df_CWE_NVD


# In[214]:


#df_CWE_MITRE, df_CWE_NVD = processAndSaveCWE(config, True)


# In[215]:


#df_CWE_MITRE
#df_CWE_NVD


# In[216]:


def load_preprocessed(config, LOAD_SAVED=True):
    
    df_CVE=processAndSaveCVE(config, LOAD_SAVED)
    df_CWE_MITRE, df_CWE_NVD = processAndSaveCWE(config, LOAD_SAVED=True)
    
    index1= np.argwhere(df_CWE_NVD['Name'].values == 'NVD-CWE-Other')[0][0]
    index2= np.argwhere(df_CWE_NVD['Name'].values == 'NVD-CWE-noinfo')[0][0]
    
    df_CWE_NVD.drop(index=[index1,index2], inplace = True)
    
    return df_CVE, df_CWE_NVD, df_CWE_MITRE


# In[217]:


#load_preprocessed(config, LOAD_SAVED=False)


# ### Create Training and Test Dataset

# In[218]:


def getMask(config,df_CVE,df_CWE):
    
    n = len(df_CWE)
    m = len(df_CVE)
    
    #get date range
    train_start_date = pd.to_datetime(str(config.TRAIN_YEARS[0])+'-01-01').tz_localize('US/Eastern')
    train_end_date = pd.to_datetime(str(config.TRAIN_YEARS[-1])+'-01-01').tz_localize('US/Eastern') + timedelta(days=365)
    val_start_date = pd.to_datetime(str(config.VAL_YEARS[0])+'-01-01').tz_localize('US/Eastern')
    val_end_date = pd.to_datetime(str(config.VAL_YEARS[-1])+'-01-01').tz_localize('US/Eastern') + timedelta(days=365)
    test_start_date = pd.to_datetime(str(config.TEST_YEARS[0])+'-01-01').tz_localize('US/Eastern')
    test_end_date = pd.to_datetime(str(config.TEST_YEARS[-1])+'-01-01').tz_localize('US/Eastern') + timedelta(days=365)
    
    cwe_ids=df_CWE['Name']    
    cwe_map=dict(zip(cwe_ids, list(range(n))))
    index_cwe_map = dict(zip(list(range(n)),cwe_ids))

    #creating y and finding labeled 
    y=torch.zeros((m,n),dtype=torch.long)    
    labeled_mask= torch.zeros(m, dtype=torch.bool)
    train_index = torch.zeros(m, dtype=torch.bool)
    test_index = torch.zeros(m, dtype=torch.bool)
    val_index = torch.zeros(m, dtype=torch.bool)
    
    CWEs=df_CVE['CWE Code']
    Dates=df_CVE['publishedDate']
    
    for i,row in enumerate(zip(CWEs,Dates)):        
        cwes=row[0]
        date=row[1]
        
        if(type(cwes) == str):
            cwes=[cwe for cwe in cwes.strip('[]').split("'") if not (cwe==',' or cwe==', ' or cwe=='''''')]        
        
        if(type(date) == str):
            date=pd.to_datetime(date)            
        
        for cwe in cwes:
            if cwe in cwe_map:                
                y[i][cwe_map[cwe]]=1
            
        if torch.sum(y[i])>0:
            labeled_mask[i]=True            
            
            if(train_start_date<date and date<train_end_date):
                train_index[i]=True                
            elif(val_start_date<date and date<val_end_date):
                val_index[i]=True
            elif(test_start_date<date and date<test_end_date):
                test_index[i]=True
            else:                
                print(date,'-> not covered')
    
    ##convert to tensors
    data=Data(train_mask=train_index, val_mask=val_index, test_mask=test_index, y=y, num_nodes=m)
        
    return data


# In[219]:


def getPercent(data,df_CVE,df_CWE, max_data_inaclass=500):
    CWEs=df_CVE['CWE Code']
    
    train_mask= (data.train_mask == True).nonzero().flatten().numpy()
    
    CWEs_train={}
    for key in train_mask:
        cwes=CWEs[key]
        if(type(cwes) == str):
            cwes=[cwe.strip() for cwe in cwes.strip('[]').split("'") if not (cwe==',' or cwe==', ' or cwe=='''''')]        
        
        for cwe in cwes:           
            if cwe in CWEs_train: 
                CWEs_train[cwe].append(key)
            else: 
                CWEs_train[cwe]=[key]
            
    
    required_train_mask = torch.zeros(len(data.train_mask), dtype=torch.bool)
        
    for key, values in CWEs_train.items(): 
        if(len(values)<max_data_inaclass):
            required_train_mask[values]=True
        else:
            np.random.shuffle(values)
            takeamnt=max_data_inaclass
            required_train_mask[values[:takeamnt]]=True
                
    data.train_mask=required_train_mask    
    
    return data


# In[ ]:





# In[220]:


from collections import OrderedDict

def CWE_description(row):    
    return str(row['Name'])+" "+str(row['Description'])+" "+str(row['Extended Description'])+" "+str(row['Common Consequences'])

def CWE_description_NVD(row,df_CWE_Mitre):
    cwe=row['Name']
    cwe_id = int(re.findall("\d+", cwe)[0])
    
    description = df_CWE_Mitre[df_CWE_Mitre['CWE-ID'].values==cwe_id]['CVE Description'].values
    if len(description)>0:
        return description[0]
    else:
        return ''

def UpdateData(data, df_CVE, df_CWE_NVD, df_CWE_MITRE):
    
    df_CWE_MITRE['CVE Description']= df_CWE_MITRE.apply(lambda row: CWE_description(row), axis=1)   
    
    for i, row in df_CWE_NVD.iterrows():
        description=CWE_description_NVD(row, df_CWE_MITRE)
        #df_CWE_NVD.set_value(i,'CVE Description',description)
        df_CWE_NVD.at[i,'CVE Description']=description

    df_CWE_NVD['CWE Code']= df_CWE_NVD.apply(lambda row: [str(row['Name'])], axis=1)
    df_CWE_NVD=df_CWE_NVD[['CVE Description','CWE Code','Name']]
    
    df_CVE_updated = pd.concat([df_CVE,df_CWE_NVD],ignore_index=True, sort=False)

    n = len(df_CWE_NVD)  
    cwe_ids=df_CWE_NVD['Name']    
    cwe_map=dict(zip(cwe_ids, list(range(n))))
    index_cwe_map = dict(zip(list(range(n)),cwe_ids))

    class_labels=torch.zeros((n,n),dtype=torch.long)
    
    CWElist=df_CWE_NVD['Name'].tolist()
    for i,cwe in enumerate(CWElist):
        cwe_value=cwe
        class_labels[i][cwe_map[cwe_value]]=1
    
    data.y=torch.cat((data.y,class_labels),dim=0)

    class_mask=torch.cat((torch.zeros(len(data.train_mask),dtype=bool),torch.ones(len(class_labels),dtype=bool)),dim=0)
    data.class_mask=class_mask

    data.train_mask=torch.cat((data.train_mask,torch.zeros(len(class_labels),dtype=bool)),dim=0)
    data.val_mask=torch.cat((data.val_mask,torch.zeros(len(class_labels),dtype=bool)),dim=0)
    data.test_mask=torch.cat((data.test_mask,torch.zeros(len(class_labels),dtype=bool)),dim=0)

    return data, df_CVE_updated, df_CWE_NVD


# ### CWE hierarchy

# In[221]:


def cwe_child(row):
    item = str(row['Related Weaknesses'])
    cve_p = re.compile('ChildOf:CWE ID:\\d+')
    results = cve_p.findall(item)
    item=''.join(results)
    cve_p = re.compile('\\d+')
    results = cve_p.findall(item)
    results = list(map(int, results))
    results=list(OrderedDict.fromkeys(results)) #preserve order
    #results=list(set(results)) #order not preserve

    # print(str(row['CWE-ID'])+'->', end="")
    # print(results)
    if(len(results)>0):
        return results
    else:
        return [-1]

def depth_node(child_parent,node):
    
    if -1 in child_parent[node]:
        return [0]
    
    depths=[]
    
    for parent_node in child_parent[node]:
        parent_depth=depth_node(child_parent, parent_node)        
        depths.extend([x+1 for x in parent_depth])
        
    return depths
                        
def create_group(nodes,parents):
    
    child_parent=dict(zip(nodes,parents))
    depth={}

    for node, level in child_parent.items():
        depth[node]=depth_node(child_parent, node)
    
    return child_parent, depth

def save_hierarchy(config, nodes, names, child_parent):
    from graphviz import Digraph,Graph
    #save hierarchy graph
    dot = Digraph(comment='NVD Research Concepts Hierarchy',engine='dot',node_attr={'shape': 'box'})
    dot.graph_attr['rankdir'] = 'LR'
    #dot=Graph(format='png')
    
    root=1003
    dot.node(str(root), "CWE-ID " + str(root) + ":" + 'Root Node')
    
    for i in range(len(nodes)):
        dot.node(str(nodes[i]), "CWE-ID " + str(nodes[i]) + ":" + names[i])


    for cwe in nodes:
        parents=child_parent[cwe]
        
        if(parents[0]==-1):
            dot.edge(str(cwe),str(root))
            continue
            
        for p in parents:
            dot.edge(str(cwe),str(p))

    #print(dot.source)
    dot.format='pdf'
    dot.render(config.GRAPHVIZ_HIERARCHY, view=False)

def cluster_cwes(config, cwe_c,df_CWE_NVD):
    
    valid_cwes=[]
    for cwe in df_CWE_NVD['Name']:
        if cwe in ['NVD-CWE-Other','NVD-CWE-noinfo']: continue    
        cwe_id = int(re.findall("\d+", cwe)[0])
        valid_cwes.append(cwe_id)
    
    delete_indexs=[]
    for i, row in cwe_c.iterrows():
        cwe=int(row['CWE-ID'])
        if(cwe not in valid_cwes):
            delete_indexs.append(i)
    
    cwe_c.drop(delete_indexs,inplace=True)
    
    parent_columns=[]
    for i, row in cwe_c.iterrows():
        parents=cwe_child(row)
        valid_parents=[]

        for x in parents:
            if x in valid_cwes:
                valid_parents.append(x)
        
        if(len(valid_parents)==0):
            valid_parents.append(-1)
        
        parent_columns.append(valid_parents)
    
    cwe_c['parent']=parent_columns
    
    nodes=cwe_c['CWE-ID'].tolist()
    parents=cwe_c['parent'].tolist()
    
    child_parent, depth = create_group(nodes, parents)
    
    save_hierarchy(config, nodes, cwe_c['Name'].tolist(),child_parent)

    return child_parent, depth

# df_CWE_MITRE, df_CWE_NVD = processAndSaveCWE(config, LOAD_SAVED=True)
# child_parent, depth = cluster_cwes(config, df_CWE_MITRE,df_CWE_NVD)


# In[222]:


def set_heirarchy(data,df_CWE_NVD):
    from ipynb.fs.full.NVD_cwe_hierarchy import get_nvd_hierarchy
    org_child_parent, org_parent_child, org_depth = get_nvd_hierarchy()
    data.org_child_parent=org_child_parent
    data.org_parent_child=org_parent_child
    data.org_depth=org_depth

    n = len(df_CWE_NVD)    
    cwe_ids=df_CWE_NVD['Name'].tolist()
    
    if 'NVD-CWE-noinfo' in cwe_ids: cwe_ids.remove('NVD-CWE-noinfo')
    if 'NVD-CWE-Other' in cwe_ids: cwe_ids.remove('NVD-CWE-Other')
    
    cwe_ids=[int(re.findall("\d+", cwe)[0]) for cwe in cwe_ids]
    
    cwe_map=dict(zip(cwe_ids, list(range(n))))
    index_cwe_map = dict(zip(list(range(n)),cwe_ids))
    
    child_parent={}
    for c,p in org_child_parent.items():
        if -1 in p:
            child_parent[cwe_map[c]]=[-1 for px in p]
        else:    
            child_parent[cwe_map[c]]=[cwe_map[px] for px in p]
    
    parent_child={}
    for p,c in org_parent_child.items():
        if p==-1:
            parent_child[-1]=[cwe_map[cx] for cx in c]
        else:            
            parent_child[cwe_map[p]]=[cwe_map[cx] for cx in c]
        
    depth={}
    for i,d in org_depth.items():
        depth[cwe_map[i]]=d
        
    data.child_parent=child_parent
    data.parent_child=parent_child
    data.depth=depth
    
    return data


# df_CWE_MITRE, df_CWE_NVD = processAndSaveCWE(config, LOAD_SAVED=True)
# set_heirarchy(Data(),df_CWE_NVD)


# In[223]:


def set_mitre_heirarchy(config, data, df_CWE_MITRE, df_CWE_NVD):
    
    org_child_parent, org_depth = cluster_cwes(config, df_CWE_MITRE,df_CWE_NVD)
    
    org_parent_child={}
    
    for c,p in org_child_parent.items():
        for px in p:                
            if px in org_parent_child:
                org_parent_child[px].append(c)
            else:
                org_parent_child[px]=[c]
    
    data.org_child_parent=org_child_parent
    data.org_parent_child=org_parent_child
    data.org_depth=org_depth

    n = len(df_CWE_NVD)    
    cwe_ids=df_CWE_NVD['Name'].tolist()
    
    if 'NVD-CWE-noinfo' in cwe_ids: cwe_ids.remove('NVD-CWE-noinfo')
    if 'NVD-CWE-Other' in cwe_ids: cwe_ids.remove('NVD-CWE-Other')
    
    cwe_ids=[int(re.findall("\d+", cwe)[0]) for cwe in cwe_ids]
    
    cwe_map=dict(zip(cwe_ids, list(range(n))))
    index_cwe_map = dict(zip(list(range(n)),cwe_ids))
    
    child_parent={}
    for c,p in org_child_parent.items():
        if -1 in p:
            child_parent[cwe_map[c]]=[-1 for px in p]
        else:    
            child_parent[cwe_map[c]]=[cwe_map[px] for px in p]
    
    parent_child={}
    for p,c in org_parent_child.items():
        if p==-1:
            parent_child[-1]=[cwe_map[cx] for cx in c]
        else:            
            parent_child[cwe_map[p]]=[cwe_map[cx] for cx in c]
        
    depth={}
    for i,d in org_depth.items():
        depth[cwe_map[i]]=d
        
    data.child_parent=child_parent
    data.parent_child=parent_child
    data.depth=depth
    
    return data

# if os.uname()[0].find('Darwin')==-1: ##if not darwin(mac/locallaptop)
#     DIR='/scratch/gilbreth/das90/Dataset'
# else:
#     DIR='/Users/siddharthashankardas/Purdue/Dataset'  

# config=Config(dataset_dir=DIR,results_dir=DIR)
# df_CWE_MITRE, df_CWE_NVD = processAndSaveCWE(config, LOAD_SAVED=True)
# set_mitre_heirarchy(config, Data(),df_CWE_MITRE, df_CWE_NVD)


# ### Dataset function

# In[224]:


def get_labeled_only(config,df_CVE,df_CWE):
    
    n = len(df_CWE)
    m = len(df_CVE)
        
    cwe_ids=df_CWE['Name']    
    cwe_map=dict(zip(cwe_ids, list(range(n))))
    labeled_mask= torch.zeros(m, dtype=torch.bool)
    
    CWEs=df_CVE['CWE Code']
    Dates=df_CVE['publishedDate']
    
    for i,row in enumerate(zip(CWEs,Dates)):        
        cwes=row[0]
        date=row[1]
        
        if(type(cwes) == str):
            cwes=[cwe for cwe in cwes.strip('[]').split("'") if not (cwe==',' or cwe==', ' or cwe=='''''')]        
                 
        
        for cwe in cwes:
            if cwe in cwe_map:                
                labeled_mask[i]=True
                break
    
    print(sum(labeled_mask))
    
    labeled_indexs= (labeled_mask == True).nonzero().flatten().numpy()
    df_CVE=df_CVE.iloc[labeled_indexs,:]
    
    return df_CVE

def take_subset(config,df_CVE,df_CWE_NVD,take=100):
    df_CVE=get_labeled_only(config,df_CVE,df_CWE_NVD)
    df_CVE=df_CVE.sample(n = take, replace = False)
    
    return df_CVE
    


# In[225]:


def ExtractFeature(config, max_data_inaclass=-1, RECOMPUTE=False, LOAD_SAVED=True, take=-1, hierarchy=''):
    
    save_filename=config.MASK_FILE
    nvd_filename=config.MERGED_NVD_CVE_FILE
    cwe_filename=config.FILTERED_NVD_CWE_FILE
    
    if RECOMPUTE==True or not os.path.exists(save_filename) or not os.path.exists(nvd_filename) or not os.path.exists(cwe_filename):
        
        df_CVE, df_CWE_NVD, df_CWE_MITRE =load_preprocessed(config, LOAD_SAVED)
        
        if take>-1: 
            print('Selecting a subset')            
            df_CVE=take_subset(config,df_CVE,df_CWE_NVD,take)
            print('Done...')
        
        data=getMask(config,df_CVE,df_CWE_NVD)
                
        if(max_data_inaclass!=-1): 
            data=getPercent(data,df_CVE,df_CWE_NVD,max_data_inaclass)
        
        data, df_CVE_merged, df_CWE = UpdateData(data, df_CVE, df_CWE_NVD, df_CWE_MITRE)
        
        if hierarchy=='nvd':
            print('Using nvd hierarchy')
            data=set_heirarchy(data,df_CWE_NVD)            
        else:
            print('using mitre hierarchy')
            data=set_mitre_heirarchy(config, data, df_CWE_MITRE, df_CWE_NVD)
        
        
    
        pickle.dump(data,open(save_filename, "wb" ))
        df_CVE_merged.to_csv(nvd_filename,index=False)
        df_CWE.to_csv(cwe_filename,index=False)
        
    else:
        data= pickle.load(open(save_filename, "rb" ))
        df_CVE_merged=pd.read_csv(nvd_filename,low_memory=False)
        df_CWE=pd.read_csv(cwe_filename,low_memory=False)
        
    return data, df_CVE_merged, df_CWE


# In[226]:


# df_CVE, df_CWE_NVD, df_CWE_MITRE = load_preprocessed(config, True)
# data=getMask(config,df_CVE,df_CWE_NVD)
# data, df_CVE_merged, df_CWE = UpdateData(data, df_CVE, df_CWE_NVD, df_CWE_MITRE)


# ## Scratch

# In[227]:


# if os.uname()[0].find('Darwin')==-1: ##if not darwin(mac/locallaptop)
#     DIR='/scratch/gilbreth/das90/Dataset'
# else:
#     DIR='/Users/siddharthashankardas/Purdue/Dataset'  

#config=Config(dataset_dir=DIR,results_dir=DIR)

data, df_CVE, df_CWE=ExtractFeature(config,max_data_inaclass=-1, RECOMPUTE=True, LOAD_SAVED=False, take=-1, hierarchy='mitre')


# In[228]:


print("Train size:", sum(data.train_mask))
print("Val size:", sum(data.val_mask))
print("Test size:", sum(data.test_mask))
print("Class size:",sum(data.class_mask))


# In[229]:


print(len(data.train_mask))


# In[230]:


# df_CVE, df_CWE_NVD, df_CWE_MITRE =load_preprocessed(config, LOAD_SAVED=True)
# take_subset(config,df_CVE,df_CWE_NVD, take=10000)


# ## Main

# In[231]:


# if __name__ == '__main__':
    
#     if os.uname()[0].find('Darwin')==-1: ##if not darwin(mac/locallaptop)
#         DIR='/scratch/gilbreth/das90/Dataset'
#     else:
#         DIR='/Users/siddharthashankardas/Purdue/Dataset'  

#     config=Config(dataset_dir=DIR,results_dir=DIR)
#     data, df_CVE, df_CWE = ExtractFeature(config,max_data_inaclass=-1, RECOMPUTE=True, LOAD_SAVED=True)
    
#     print("Train size:", sum(data.train_mask))
#     print("Val size:", sum(data.val_mask))
#     print("Test size:", sum(data.test_mask))
#     print("Class size:",sum(data.class_mask))
    
#     print("Total: ",sum(data.train_mask)+sum(data.val_mask)+sum(data.test_mask))


# In[ ]:




