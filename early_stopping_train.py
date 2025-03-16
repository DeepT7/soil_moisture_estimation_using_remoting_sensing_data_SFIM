import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from model import RegressionANN, RMSELoss, EarlyStopping, plot_metrics

# Training loop
model = RegressionANN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# criterion = torch.nn.MSELoss()
criterion = RMSELoss()
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min',  patience = 100, factor = 0.1, threshold= 0.0001)
early_stopping = EarlyStopping(patience=1000, min_delta=0.0001, path="models/best_model.pt")

# Chuẩn hóa dữ liệu
data = pd.read_csv('normalized.csv')
numeric_cols = ["Sentinel-1 VH", "SMAP", "NDVI", "LST", "sm"]
data[numeric_cols] = data[numeric_cols].apply(pd.to_numeric, errors = 'coerce')
data = data.dropna()
X = data.iloc[:, 4:8].values
y = data.iloc[:,8].values.reshape(-1, 1)

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size = 0.4, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

batch_size = 8
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle = False)

print(f'Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}')

model.load_state_dict(torch.load('models/best_model_1.pt'))   
model.eval()

with torch.no_grad():
    y_pred = model(X_test_tensor)

mse = nn.MSELoss()
mse_loss = mse(y_pred, y_test_tensor)
print('mse_loss', mse_loss)
# Convert tensors to NumPy arrays
y_pred_np = y_pred.numpy().flatten()  # Flatten to 1D
y_test_np = y_test_tensor.numpy().flatten()

# Print side-by-side
print("Predicted  |  Actual")
print("---------------------")
for pred, actual in zip(y_pred_np, y_test_np):
    print(f"{pred:.4f}    |  {actual:.4f}")

# Chuyển sang tensor
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# for epoch in range(5000):
#     # Forward pass
#     pred = model(X_tensor)
#     loss = criterion(pred, y_tensor)
    
#     # Backpropagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
    
#     if (epoch+1) % 10 == 0:
#         print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')

####################################################################3
# train_losses = []
# val_losses = []
# for epoch in range(5000):
#     model.train()  # Set model to training mode
    
#     # Forward pass
#     train_pred = model(X_train_tensor)
#     loss = criterion(train_pred, y_train_tensor)
    
#     # Backpropagation
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     train_loss = loss.item()
#     train_losses.append(train_loss)
    
#     # Validation Phase
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     with torch.no_grad():  # No need to compute gradients during validation
#         val_pred = model(X_val_tensor)
#         val_loss = criterion(val_pred, y_val_tensor)

#         val_loss = val_loss.item()
#         val_losses.append(val_loss)

#     # Reduce learning rate if val loss didn't improve
#     scheduler.step(val_loss)

#     if early_stopping(val_loss, model):
#         break

#     # Print training and validation loss every 10 epochs
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# plot_metrics(train_losses,val_losses, 'rmse_3_layers')

###############################################################################

# for epoch in range(5000):
#     model.train()  # Set model to training mode
#     train_loss = 0.0
    
#     for batch_X, batch_y in train_loader:
#         # Forward pass
#         pred = model(batch_X)
#         loss = criterion(pred, batch_y)
        
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         train_loss += loss.item()
    
#     # Compute average training loss
#     train_loss /= len(train_loader)
    
#     # Validation Phase
#     model.eval()  # Set model to evaluation mode
#     val_loss = 0.0
#     with torch.no_grad():  # No need to compute gradients during validation
#         for val_X, val_y in val_loader:
#             val_pred = model(val_X)
#             loss = criterion(val_pred, val_y)
#             val_loss += loss.item()
    
#     val_loss /= len(val_loader)

#     # Print training and validation loss every 10 epochs
#     if (epoch+1) % 10 == 0:
#         print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")



