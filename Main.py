from classes.dataloader import DatasetLoader
from classes.cat_norm import Cat_Norm
from classes.correlation import CorrelationMatrix
from classes.datasplitter import DataSplitter
from classes.autoencoder import Autoencoder
from classes.train import Trainer
from classes.predict import Predictor
from classes.anomaly_detect import AnomalyDetector, plot_anomaly
from classes.encoder import Encoder
from classes.decoder import Decoder

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import logging
import warnings
import copy
import os


logging.basicConfig(level=logging.INFO, 
                    filename='logs/main.log', 
                    filemode='w', 
                    format='%(name)s - %(levelname)s - %(message)s')


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

# correlation matrix
corr_mat = CorrelationMatrix(df_encoded)
corr_mat.plot_correlation_heatmap()
corr_mat.remove_highly_correlated_variables
corr_mat.plot_correlation_selected_vars()

# normal data (label = 0) vs anomalies (label != 0)
df_normal = df_encoded[df_encoded['attack_cat'] == 6]
df_anomalies = df_encoded[df_encoded['attack_cat'] != 6]
df_anomalies = torch.as_tensor(df_anomalies.to_numpy(), dtype=torch.float)

# train/test data split
data_splitter = DataSplitter(df_normal, test_size = 0.3, random_state = 42)
X_train, X_test, Y_test = data_splitter.split_data()


# parameters
input_size = X_train.shape[1]
output_size = X_train.shape[1]
hidden_size1 = 28
hidden_size2 = 15
center_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 25
batch_size = 16
lr = 0.01

# model train
model = Autoencoder(input_size, output_size, hidden_size1, hidden_size2, center_size).to('cpu')
trainer = Trainer(model, X_train, X_test, n_epochs, batch_size, lr, device)
trained_model, history = trainer.train_model()
trainer.plot_loss_history(history)

# prediction
predictor = Predictor(model, device)
prediction_train, losses_train = predictor.predict(batch_size, X_train)
prediction_test, losses_test = predictor.predict(batch_size, X_test)
prediction_anomalies, losses_anomalies = predictor.predict(batch_size, df_anomalies)
predictor.plot_loss_distributions(losses_train, losses_test, losses_anomalies)

# anomaly detection 
detector = AnomalyDetector(threshold = 500)
classifications = detector.detect_anomalies(losses_anomalies)
visualizer = plot_anomaly(losses_train, losses_test, losses_anomalies)
visualizer.plot_losses()
