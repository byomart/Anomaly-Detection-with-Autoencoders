import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

class CorrelationMatrix:
    def __init__(self, df):
        self.df = df

    def plot_correlation_heatmap(self):
        sns.heatmap(self.df.corr())
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join('images', 'correlation_heatmap.png')) 
        # plt.show()
        plt.close()

    def remove_highly_correlated_variables(self, threshold=0.97):
        correlated_vars = []
        corr = self.df.corr()
        columns = self.df.columns.tolist()
        for i in range(len(columns) - 1):
            for j in range(i+1, len(columns)):
                if corr[columns[i]][columns[j]] > threshold:
                    correlated_vars.append(columns[j])

        self.df = self.df.drop(columns=correlated_vars)
        return self.df

    def plot_correlation_selected_vars(self, k=10):
        corrmat = self.df.corr()
        cols = corrmat.nlargest(k, 'attack_cat')['attack_cat'].index
        cm = np.corrcoef(self.df[cols].values.T)

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        sns.heatmap(corrmat, vmax=.8, square=True)
        plt.title('"attack_cat" correlation matrix')

        plt.subplot(1, 2, 2)
        sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 5}, yticklabels=cols.values, xticklabels=cols.values)
        plt.title('"attack_cat" 10 most correlated')
        
        plt.tight_layout()
        plt.savefig(os.path.join('images', 'attack_cat_correlation_matrix.png'))
        plt.close()
