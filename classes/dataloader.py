import pandas as pd
import logging
import matplotlib.pyplot as plt
import os


class DatasetLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_dataset(self):
        try:
            self.data = pd.read_csv(self.file_path)
            print("Dataset loaded successfully.")
            logging.info('loaded df')
            logging.info(self.data.head())
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def get_data(self):
        return self.data
    
    def log_dataframe_info(self, data):
        head = self.data.head().columns.tolist()
        logging.info("Number of columns: {0}\nNumber of rows: {1}\nName of the first column: {2}\nName of the last column: {3}".format(len(head), self.data.size, head[0], head[-1]))
        logging.info("Types of detected attacks:\n {0}".format(' '.join(self.data[head[-2]].unique())))
        logging.info("Different elements in the last column: {0}".format(self.data[head[-1]].unique()))
        logging.info("Exist Null elements: {0}\n".format(any(self.data.isna().sum() > 0)))
        logging.info(self.data['label'].value_counts())

    def log_attack_types(self, data):
        df_anomaly = data.loc[data['label'] == 1]
        df_not_anomaly = data.loc[data['label'] == 0]

        logging.info("Types of attacks detected among anomalies:")
        logging.info(df_anomaly['attack_cat'].value_counts())

        logging.info("Types of attacks detected among non-anomalies:")
        logging.info(df_not_anomaly['attack_cat'].value_counts())


    def attack_value_count(self, data):
        vc = data['attack_cat'].value_counts(ascending=False)
        explode = [0.0] * len(vc)
        plt.figure(figsize=(18, 18))
        plt.pie(x=vc.values, labels=vc.index, explode=explode, autopct='%1.1f%%')
        plt.savefig(os.path.join('images', 'attack_pie_chart.png'))
        plt.close()




