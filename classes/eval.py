import numpy as np
import matplotlib.pyplot as plt
import os

class ConfusionMatrixCalculator:
    def __init__(self, threshold):
        self.threshold = threshold

    def calculate_confusion_matrix(self, losses_normal, losses_anomalies):
        all_normal = np.array(losses_normal)
        all_anomalies = np.array(losses_anomalies)
        
        TN = np.sum(all_normal <= self.threshold)
        FP = np.sum(all_normal > self.threshold)
        TP = np.sum(all_anomalies > self.threshold)
        FN = np.sum(all_anomalies <= self.threshold)

        accuracy = (TP + TN) / (TP + TN + FP + FN)

        return accuracy, TP, FN, TN, FP

    
    def plot_confusion_matrix(self, TP, FN, TN, FP):
            confusion_matrix = np.array([[TP, FP],
                                        [FN, TN]])

            plt.figure()
            plt.imshow(confusion_matrix, cmap='Blues')
            plt.xticks(np.arange(2), ['Positive', 'Negative'])
            plt.yticks(np.arange(2), ['Positive', 'Negative'])
            plt.xlabel('Predicted label')
            plt.ylabel('True label')
            for i in range(2):
                for j in range(2):
                    plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')
            plt.title('Confusion Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join('images', 'confusion_mat.png'))
            plt.show()