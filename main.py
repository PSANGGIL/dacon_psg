from db import DB
from torch.utils.data import DataLoader, Dataset
from model import *
from util._result import *
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import pandas as pd
if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#    #foece
#    epochs =100 
#    
#    input_size = 3
#    h1_size = 64
#    h2_size = 128
#    h3_size = 64
#    learning_rate = 0.01 
#    batch_size = 8192
#    
#    train_dataset = DB('./data/Train.xyz', 'force')
#    test_dataset = DB('./data/Test.xyz', 'force')
#    
#    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
#    test_loader =  DataLoader(test_dataset,  batch_size= 2 * batch_size, shuffle=False, num_workers=1)
#    
#    model = ForceModel(batch_size, input_size, h1_size, h2_size, h3_size).to(device)
#    criterion = nn.MSELoss()
#    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
#
#    result = force_submission_data( 
#                            force_trainer(epochs, model, train_loader, optimizer, criterion, device),
#                            test_force(model, test_loader, device ))
#    result.to_csv('v1.csv', index = False)
#    
#
    #energy
    epochs =0 
   
    input_size = 4  # feature 개수
    hidden_size = 256
    output_size = 1 # target 개수
    batch_size = 8192
    learning_rate = 0.01
 
    
    train_dataset = DB('./data/Train.xyz', 'energy')
    test_dataset = DB('./data/Test.xyz'  , 'energy')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    test_loader =  DataLoader(test_dataset,  batch_size= 2 * batch_size, shuffle=False, num_workers=1)
    
    
    model = EnergyModel(batch_size, input_size, hidden_size, num_layers=1, dropout_rate=0.5).to(device)
    
    criterion = nn.GaussianNLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
       

    train_energy(epochs, model, train_loader, optimizer, criterion, device) 

    test_energy(model, test_loader, device) 
    exit(-1)

    result = force_submission_data( 
                            force_trainer(epochs, model, train_loader, optimizer, criterion, device),
                            test_force(model, test_loader, device ))
    result.to_csv('v1.csv', index = False)

