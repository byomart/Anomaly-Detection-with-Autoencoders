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
from classes.threshold import AutomaticThresholdDetector
from classes.eval import ConfusionMatrixCalculator

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
loader.log_dataframe_info(df)
loader.log_attack_types(df)
loader.attack_value_count(df)

df = df.drop(columns=['id'])

# numerical columns categorization and normalization
cat_norm = Cat_Norm()
df = cat_norm.Cat(df)  
df_encoded = cat_norm.Norm(df)  

# correlation matrix
corr_mat = CorrelationMatrix(df_encoded)
corr_mat.plot_correlation_heatmap()
corr_mat.remove_highly_correlated_variables
corr_mat.plot_correlation_selected_vars()

# normal data (label = 0) vs anomalies (label != 0)
df_normal = df_encoded[df_encoded['attack_cat'] == 6]
df_normal_tensor = torch.as_tensor(df_normal.to_numpy(), dtype=torch.float)
df_anomalies = df_encoded[df_encoded['attack_cat'] != 6]
df_anomalies_tensor = torch.as_tensor(df_anomalies.to_numpy(), dtype=torch.float)

# train/test data split
data_splitter = DataSplitter(df_normal, test_size = 0.3, random_state = 42)
X_train, X_test, Y_test = data_splitter.split_data()


# parameters
input_size = X_train.shape[1]
output_size = X_train.shape[1]
hidden_size1 = 12
hidden_size2 = 7
center_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_epochs = 30
batch_size = 12
lr = 0.005

# # model train
# model = Autoencoder(input_size, output_size, hidden_size1, hidden_size2, center_size).to('cpu')
# trainer = Trainer(model, X_train, X_test, n_epochs, batch_size, lr, device)
# trained_model, history = trainer.train_model()
# trainer.plot_loss_history(history)
model = torch.load("model.pth")


# prediction
predictor = Predictor(model, device)
prediction_normal, losses_normal = predictor.predict(df_normal_tensor)
prediction_anomalies, losses_anomalies = predictor.predict(df_anomalies_tensor)
predictor.plot_loss_distributions(losses_normal, losses_anomalies)

# automatic th detection
th_detector = AutomaticThresholdDetector()
threshold = th_detector.detect_threshold(losses_normal, losses_anomalies)


confusion_matrix_calculator = ConfusionMatrixCalculator(threshold)
accuracy,TP, FN, TN, FP = confusion_matrix_calculator.calculate_confusion_matrix(losses_normal, losses_anomalies)
confusion_matrix_calculator.plot_confusion_matrix(TP, FN, TN, FP)
logging.info(f'Accuracy: {accuracy}, True Positives: {TP}, False Positives: {FP}, True Negatives: {TN}, False Negatives: {FN} ')


'''
all_attack = np.array(losses_anomalies)
TP = all_attack[(all_attack > threshold)]
FN = all_attack[(all_attack <= threshold)]

all_normal = np.array(losses_normal)
FP = all_normal[(all_normal > threshold)]
TN = all_normal[(all_normal <= threshold)]

print(f'True Positives: {len(TP)}')
print(f'False Negatives: {len(FN)}')
print(f'True Positive Rate / Sensitivity: {len(TP)/(len(TP)+len(FN))}')
print('\n')
print(f'True Negatives: {len(TN)}')
print(f'False Positives: {len(FP)}')
print(f'True Negative Rate / Specificity: {len(TN)/(len(TN)+len(FP))}')
print('\n')


anomaly_tag={
    'Analysis': 0,
    'Backdoor': 1,
    'DoS': 2,
    'Exploits': 3,
    'Fuzzers': 4,
    'Generic': 5,
    'Reconnaissance': 7,
    'Shellcode': 8,
    'Worms': 9
             }


g_data=analyzer.print_data_distribution(losses_normal, losses_anomalies, "Ditrubución global de los datos")

thrs_in_anomaly={}
thrs_in_anomaly['General']=g_data


attack_analyzer = AttackDistributionAnalyzer(model,device)
thrs_in_anomaly = attack_analyzer.analyze_and_update(thrs_in_anomaly, df_anomalies, anomaly_tag, losses_normal)'''