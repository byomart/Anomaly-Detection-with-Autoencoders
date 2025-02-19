import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrixPlotter:
    def plot_confusion_matrix(self, TP, FN, TN, FP):
        confusion_matrix = np.array([[TP, FP],
                                      [FN, TN]])

        fig, ax = plt.subplots()
        im = ax.imshow(confusion_matrix, cmap='Blues')

        ax.set_xticks(np.arange(2))
        ax.set_yticks(np.arange(2))
        ax.set_xticklabels(['Positive', 'Negative'])
        ax.set_yticklabels(['Positive', 'Negative'])

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for i in range(2):
            for j in range(2):
                text = ax.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

        ax.set_title('Confusion Matrix')
        fig.tight_layout()
        plt.show()