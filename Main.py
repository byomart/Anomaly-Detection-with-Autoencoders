import os
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import TensorDataset, WeightedRandomSampler
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.gridspec as gridspec
import warnings
from sklearn.model_selection import train_test_split
import copy
from DataLoader import DatasetLoader
from Cat_Norm import Cat_Norm
import logging

logging.basicConfig(level=logging.INFO, filename='main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


# dataset load
file_path = "data/UNSW_NB15_training-set.csv" 
loader = DatasetLoader(file_path)
loader.load_dataset()
df = loader.get_data()
df = df.drop(columns=['id'])
logging.info('loaded df')
logging.info(df.head())


# numerical columns categorization and normalization
cat_norm = Cat_Norm()
cat_norm.Cat(df)  
df_encoded = cat_norm.Norm(df)  
logging.info('encoded and nomralized df')
logging.info(df_encoded.head())




