import torch
import torch.nn as nn
import numpy as np
import copy
import matplotlib.pyplot as plt
import os
import logging

class Trainer:
    def __init__(self, model, train_dataset, val_dataset, n_epochs, batch_size, lr, device):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device

    def train_model(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)
        criterion = nn.L1Loss(reduction='sum').to(self.device)
        history = dict(train=[], val=[])

        best_model_wts = copy.deepcopy(self.model.state_dict())
        best_loss = float('inf')

        for epoch in range(1, self.n_epochs + 1):
            train_losses = []
            self.model.train()
            for i in range(0, len(self.train_dataset), self.batch_size):
                seq_true = self.train_dataset[i:i + self.batch_size].to(self.device)
                seq_pred = self.model(seq_true)

                loss = criterion(seq_pred, seq_true)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            self.model.eval()
            with torch.no_grad():
                for i in range(0, len(self.val_dataset), self.batch_size):
                    seq_true = self.val_dataset[i:i + self.batch_size].to(self.device)
                    seq_pred = self.model(seq_true)

                    loss = criterion(seq_pred, seq_true)
                    val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            scheduler.step(val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(self.model.state_dict())
                
            print(f'Epoch {epoch}: train_loss: {train_loss:.8f}\t val_loss: {val_loss:.8f}')
            logging.info(f'Epoch {epoch}: train_loss: {train_loss:.8f}\t val_loss: {val_loss:.8f}')
            
        self.model.load_state_dict(best_model_wts)

        torch.save(self.model, 'model.pth')
        return self.model.eval(), history
        
    def plot_loss_history(self, history):
        plt.plot(history['train'][:])
        plt.plot(history['val'][:])
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'val'])
        plt.title('Loss over training epochs')  
        plt.savefig(os.path.join('images', 'loss.png'))
        plt.close() 
        
        
