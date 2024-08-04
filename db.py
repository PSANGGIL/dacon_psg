import torch
import pandas as pd
import numpy as np
import math
import json
import h5py
from tqdm import tqdm, trange
from glob import glob
from time import time
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

#from rdkit import Chem
#from rdkit.Chem import rdchem
#from rdkit.Geometry import Point3D

import os
from util._xyz2h5 import *

def Normalization(data):
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data, min_val, max_val

class DB(Dataset):
    def __init__(self, filename, target ,precision=32):
        if precision==16:
            self.precision = torch.half
        elif precision==32:
            self.precision = torch.float32
        elif precision==64:
            self.precision = torch.float64

        #with open('factor.json', 'w') as f:
        #    json.dump({'min_val': float(self.energy[1]), 'max_val': float(self.energy[2])}, f)
        
        self.filename = filename
        st = time()        
        job_type = os.path.splitext(os.path.basename(filename))[0]
        
        if not os.path.exists('./data/' + job_type + '_db.h5'):
            make_custum_dataset(filename) 
        else:
            pass
        
        self.data = load_h5('./data/' + job_type + '_db.h5')
        self.target = target
       
        if self.target == 'force':
            self.force_atoms   = torch.tensor(self.data['atoms'].reshape(-1), dtype = torch.int)
            self.force_x       = torch.tensor(self.data['x'].reshape(-1)      , dtype = self.precision) 
            self.force_y       = torch.tensor(self.data['y'].reshape(-1)      , dtype = self.precision) 
            self.force_z       = torch.tensor(self.data['z'].reshape(-1)      , dtype = self.precision) 
            self.force_fx      = torch.tensor(self.data['fx'].reshape(-1)     , dtype = self.precision) 
            self.force_fy      = torch.tensor(self.data['fy'].reshape(-1)     , dtype = self.precision) 
            self.force_fz      = torch.tensor(self.data['fz'].reshape(-1)     , dtype = self.precision) 
            self.force_energy  = torch.repeat_interleave(torch.tensor(self.data['energy'].reshape(-1), dtype = self.precision), len(self.data['x']))
            self.force_force_feature  = torch.tensor(self.data['force_features'].reshape(-1,3) , dtype = self.precision)       
            self.force_energy_feature  = torch.repeat_interleave(torch.tensor(self.data['energy_features'].reshape(-1), dtype = self.precision), len(self.data['x']))
            self.atomic_energy_feature = torch.tensor(self.data['atomic_features'].reshape(-1)     , dtype = self.precision) 

        print(time() - st)
    def __len__(self):
        if self.target == 'energy':
            return len(self.data['x'])
        
        elif self.target == 'force':
            return len(self.data['x'].reshape(-1))

    def __getitem__(self, idx):
        if self.target == 'energy':
            atoms   = torch.tensor(self.data['atoms'][idx,:], dtype = torch.int)
            x       = torch.tensor(self.data['x'][idx, :]   , dtype = self.precision) 
            y       = torch.tensor(self.data['y'][idx, :]   , dtype = self.precision) 
            z       = torch.tensor(self.data['z'][idx, :]   , dtype = self.precision) 
            fx      = torch.tensor(self.data['fx'][idx,:]   , dtype = self.precision) 
            fy      = torch.tensor(self.data['fy'][idx,:]   , dtype = self.precision) 
            fz      = torch.tensor(self.data['fz'][idx,:]   , dtype = self.precision) 
            energy  = torch.tensor(self.data['energy'][idx,:], dtype = self.precision) 
            force_feature  = torch.tensor(self.data['force_features'][idx, :, :] , dtype = self.precision)       
            energy_feature = torch.tensor(self.data['energy_features'][idx]      , dtype = self.precision)       
            atomic_feature = torch.tensor(self.data['atomic_features'][idx, :]      , dtype = self.precision)  
        elif self.target == 'force':
            atoms           = self.force_atoms[idx]          
            x               = self.force_x[idx]     
            y               = self.force_y[idx]  
            z               = self.force_z[idx]  
            fx              = self.force_fx[idx]  
            fy              = self.force_fy[idx]  
            fz              = self.force_fz[idx]  
            energy          = self.force_energy  [idx]  
            force_feature   = self.force_force_feature[idx] 
            energy_feature  = self.force_energy_feature[idx]
            atomic_feature  = self.atomic_energy_feature[idx]
        return atoms, x, y, z, force_feature, energy_feature, fx, fy, fz, energy, atomic_feature 

if __name__ == "__main__":

    db = DB('./data/Train.xyz', 'energy')
    #db = DB('./data/Test.xyz', 'force')
    print(db.__len__())
    print(db[0])
