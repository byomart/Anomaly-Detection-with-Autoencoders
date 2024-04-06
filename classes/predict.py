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

    def predict(self, batch_size, dataset):
        predictions, losses = [], []
        criterion = nn.MSELoss(reduction='sum').to(self.device)

        with torch.no_grad():
            self.model = self.model.eval()
            for i in range(0, len(dataset), batch_size):
                seq_true = dataset[i:i + batch_size].to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)

                predictions.append(seq_pred.cpu().numpy())
                losses.append(loss.item())
        return predictions, losses
    
    def plot_loss_distributions(self, losses_train, losses_test, losses_anomalies):
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.kdeplot(losses_train, color='b')
        plt.subplot(1, 3, 2)
        sns.kdeplot(losses_test, color='r')
        plt.subplot(1, 3, 3)
        sns.kdeplot(losses_anomalies, color='g')
        plt.savefig(os.path.join('images', 'loss_distributions.png'))
        plt.close()

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        sns.kdeplot(losses_train, color='b')
        sns.kdeplot(losses_test, color='r')
        sns.kdeplot(losses_anomalies, color='g')
        plt.savefig(os.path.join('images', 'combined_loss_distributions.png'))
        plt.close()