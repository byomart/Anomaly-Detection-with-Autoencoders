import torch
from sklearn.model_selection import train_test_split
import logging

class DataSplitter:
    def __init__(self, df, test_size, random_state):
        self.df = df
        self.test_size = test_size
        self.random_state = random_state
    
    def split_data(self):
        X_train, X_test = train_test_split(self.df, test_size=self.test_size, random_state=self.random_state)
        Y_test = X_test['attack_cat']
        
        # to tensor
        X_train = torch.as_tensor(X_train.to_numpy(), dtype=torch.float)
        X_test = torch.as_tensor(X_test.to_numpy(), dtype=torch.float)
        Y_test = torch.as_tensor(Y_test.to_numpy(), dtype=torch.float)
        
        logging.info(f'Trainset: {X_train.shape}')
        logging.info(f'Testset: {X_test.shape}')
        
        return X_train, X_test, Y_test
