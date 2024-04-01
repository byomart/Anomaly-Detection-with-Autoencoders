import torch
import matplotlib.pyplot as plt
import numpy as np
import logging
import os

class AnomalyDetector:
    def __init__(self, threshold):
        self.threshold = threshold

    def detect_anomalies(self, reco_loss):
        classifications = ['Normal' if np.mean(reco_loss) <= self.threshold else 'Anómala' for reco_loss in reco_loss]
        logging.info(classifications)
        return classifications

class plot_anomaly:
    def __init__(self, losses_train, losses_test, losses_anomalies):
        self.losses_train = losses_train
        self.losses_test = losses_test
        self.losses_anomalies = losses_anomalies

    def plot_losses(self):
        plt.figure(figsize=(15, 5))
        y_min = min(min(self.losses_train), min(self.losses_test), min(self.losses_anomalies))
        y_max = max(max(self.losses_train), max(self.losses_test), max(self.losses_anomalies))

        plt.subplot(1, 3, 1)
        plt.plot(self.losses_train)
        plt.ylim(y_min, y_max)
        plt.title('Train')

        plt.subplot(1, 3, 2)
        plt.plot(self.losses_test)
        plt.ylim(y_min, y_max)
        plt.title('Test')

        plt.subplot(1, 3, 3)
        plt.plot(self.losses_anomalies)
        plt.ylim(y_min, y_max)
        plt.title('Anomalies')
        
        plt.savefig(os.path.join('images', 'anomaly_detect.png'))
        plt.close()