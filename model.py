import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import torch.nn as nn
import matplotlib.pyplot as plt

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true):
        return torch.sqrt(self.mse(y_pred, y_true))
    
class EarlyStopping:
    def __init__(self, patience = 1000, min_delta = 0, path = 'models/checkpoint.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.best_loss = np.inf
        self.counter = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss       
            self.counter = 0
            torch.save(model.state_dict(), self.path) # Save the best model

        else:
            self.counter += 1
            if self.counter  > self.patience:
                print("Early stopping !")
                return True
            
        return False
    


class RegressionANN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden1 = torch.nn.Linear(4, 2)
        self.bn1 = nn.BatchNorm1d(2)
        # self.hidden2 = torch.nn.Linear(8, 12)
        # self.bn2 = nn.BatchNorm1d(12)
        # self.hidden3 = torch.nn.Linear(12, 4)
        # self.bn3 = nn.BatchNorm1d(4)
        self.output = torch.nn.Linear(2, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn1(self.hidden1(x)))
        # x = self.relu(self.bn2(self.hidden2(x)))
        # x = self.relu(self.bn3(self.hidden3(x)))
        return self.output(x)

def plot_metrics(train_losses,val_losses, name):
    plt.figure(figsize=(8, 6))
    plt.plot(train_losses, label = 'Train_RMSE', color = 'blue')
    plt.plot(val_losses, label = 'Val_RMSE', color = 'red')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.title('Train and Validation RMSE Loss')
    plt.legend()
    plt.grid()
    plt.savefig(f'models/{name}.png')
    plt.show()
