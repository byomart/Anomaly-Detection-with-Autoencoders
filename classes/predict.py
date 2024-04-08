import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Predictor:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def predict(self, dataset):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction='sum').to(self.device)

        with torch.no_grad():
            self.model = self.model.eval()
            for i in range(dataset.shape[0]):
                seq_true = dataset[i:i+1].to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)

                predictions.append(seq_pred.cpu().numpy())
                losses.append(loss.item())
        return predictions, losses
    
    def plot_loss_distributions(self, losses_normal, losses_anomalies):
        
        plt.figure(figsize=(15, 5))
        plt.hist(losses_normal, bins=50);
        plt.hist(losses_anomalies, bins=50);
        plt.savefig(os.path.join('images', 'loss_distributions_hist.png'))
        plt.close()
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        sns.kdeplot(losses_normal, color='b')
        plt.subplot(1, 2, 2)
        sns.kdeplot(losses_anomalies, color='r')
        plt.savefig(os.path.join('images', 'loss_distributions.png'))
        plt.close()

        plt.figure(figsize=(15, 5))
        sns.kdeplot(losses_normal, color='b')
        sns.kdeplot(losses_anomalies, color='r')
        plt.axvline(x = 2, color='k', linestyle='--', label='Threshold')
        plt.savefig(os.path.join('images', 'combined_loss_distributions.png'))
        plt.close()