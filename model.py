import re
import json
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

class GNNForceFieldPredictor(torch.nn.Module):
    def __init__(self):
        super(GNNForceFieldPredictor, self).__init__()
        self.conv1 = GCNConv(3, 64)  # 입력 노드 특성은 XYZ 좌표
        self.conv2 = GCNConv(64, 64)
        self.lin = torch.nn.Linear(64, 3)  # 출력은 3차원 벡터 (힘)

    def forward(self, node_index, edge_index):
        x, edge_index = node_index, edge_index
        edges = torch.tensor([[i, (i + 1) % x.size(1)] for i in range(x.size(1))] * x.size(0), dtype=torch.long).t().to(device)
        batch = torch.tensor([i // x.size(1) for i in range(x.size(0) * x.size(1))], dtype=torch.long).reshape(x.size(0), x.size(1))
        batch = torch.tensor([i // x.size(1) for i in range(x.size(0) * x.size(1))], dtype=torch.long)
        x = self.conv1(x, edges)
        x = F.relu(x)
        x = self.conv2(x, edges)
        x = F.relu(x)
        x = self.lin(x)
        return x

class ForceModel(nn.Module):
    def __init__(self, batch_size, input_size, h1_size, h2_size, h3_size):
        super(ForceModel, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, h1_size),
            nn.BatchNorm1d(h1_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(h1_size, h2_size),
            nn.BatchNorm1d(h2_size),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.5),
            
            nn.Linear(h2_size, h3_size),
            nn.BatchNorm1d(h3_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(h3_size, 3)
        )
        self.sum_features = nn.Linear(input_size * 2, 3)
    def forward(self, x, features):
        y1 = self.layers(x)
        y2 = self.layers(features)
        y_ = self.sum_features(torch.cat((y1, y2), dim = -1))
        return y_


class EnergyModel(nn.Module):
    def __init__(self, batch_size, input_size, hidden_size, num_layers=1, dropout_rate=0.5):
        super(EnergyModel, self).__init__()
        
        # Bidirectional LSTM with Dropout
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, 
                            dropout=dropout_rate,
                            bidirectional=True)
        
        # Bidirectional LSTM이므로 hidden_size 조정
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2_mean = nn.Linear(hidden_size, 1)
        self.fc2_variance = nn.Linear(hidden_size, 1)
        self.relu = nn.ReLU()
        self.batchnorm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softplus = nn.Softplus()  
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        x = self.fc1(lstm_out[:, -1, :])
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.dropout(x)

        mean = self.fc2_mean(x)
        variance = self.softplus(self.fc2_variance(x))
        return mean, variance


def force_trainer(epochs, model, dataloader, optimizer, criterion, device): 
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for atoms, x, y, z, force_feature, energy_feature, fx, fy, fz, energy, atomic_energy in tqdm(dataloader):
            optimizer.zero_grad()
            inputs = torch.stack([x,y,z], dim = 1).to(device) 
            labels = torch.stack([fx,fy,fz], dim = 1).to(device) 
            features = force_feature.to(device)
            outputs = model(inputs, features)
            loss = criterion(outputs, labels)
            train_loss += loss    
            loss.backward()
            optimizer.step()
        print(f"{epoch+1}/{epochs}, force_loss : {train_loss/len(dataloader)}")
    
    return (train_loss/len(dataloader)).detach().cpu().numpy()
     
def test_force(model, dataloader, device): 
    model.eval()
    preds = []
    with torch.no_grad():
        for atoms, x, y, z, force_feature, energy_feature, fx, fy, fz, energy, atomic_energy in tqdm(dataloader):
            inputs = torch.stack([x,y,z], dim = 1).to(device) 
            energy_features, 
            features = force_feature.to(device)
            
            labels = torch.stack([fx,fy,fz], dim = 1).to(device) 
            features = force_feature.to(device)
            outputs = model(inputs, features)
    
            pred = outputs.detach().cpu().numpy()
            preds.extend(pred)
    
    print("Inference Complete!")
    return preds

def train_energy(epochs, model, dataloader, optimizer, criterion, device): 
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for atoms, x, y, z, force_feature, energy_feature, fx, fy, fz, energy, atomic_energy in tqdm(dataloader):
            
            inputs = torch.stack([x, y, z, atomic_energy], dim = -1).to(device) 
            labels = energy.to(device)
            
            optimizer.zero_grad()
            
            mean, variance = model(inputs)
            loss = criterion(mean, labels, variance) 
            train_loss += loss    
            loss.backward()
            optimizer.step()
            
        print(f"{epoch+1}/{epochs}, force_loss : {train_loss/len(dataloader)}")
    
def test_energy(model, dataloader, device): 
    model.eval()
    
    preds_mean = []
    preds_variance = []
    with torch.no_grad():
        for atoms, x, y, z, force_feature, energy_feature, fx, fy, fz, energy, atomic_energy in tqdm(dataloader):
            
            inputs = torch.stack([x, y, z, atomic_energy], dim = -1).to(device) 
            labels = energy.to(device)
    
            mean, variance = model(inputs)
            pred_mean = mean.detach().cpu().numpy()
            pred_variance = variance.detach().cpu().numpy()
    
            preds_mean.extend(pred_mean)
            preds_variance.extend(pred_variance)
     
    return preds_mean, preds_variance
