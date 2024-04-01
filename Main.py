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
from Correlation import CorrelationMatrix
from DataSplitter import DataSplitter
import logging

logging.basicConfig(level=logging.INFO, filename='logs/main.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# dataset load
file_path = "data/UNSW_NB15_training-set.csv" 
loader = DatasetLoader(file_path)
loader.load_dataset()
df = loader.get_data()
df = df.drop(columns=['id'])

# numerical columns categorization and normalization
cat_norm = Cat_Norm()
cat_norm.Cat(df)  
df_encoded = cat_norm.Norm(df)  
# logging.info('encoded and normalized df')
# logging.info(df_encoded.head())

# correlation matrix
corr_mat = CorrelationMatrix(df_encoded)
corr_mat.plot_correlation_heatmap()
corr_mat.remove_highly_correlated_variables
corr_mat.plot_correlation_selected_vars()

# normal data (label = 0) vs anomalies (label != 0)
df_normal = df_encoded[df_encoded['attack_cat'] == 6]
df_anomalies = df_encoded[df_encoded['attack_cat'] != 6]

# train/test data split
data_splitter = DataSplitter(df_normal, test_size = 0.3, random_state = 42)
X_train, X_test, Y_test = data_splitter.split_data()

