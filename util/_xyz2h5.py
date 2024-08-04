import torch
import os
import h5py
import pandas as pd
import numpy as np
from util._force import *

def make_custum_dataset(filename):
    job_type = os.path.splitext(os.path.basename(filename))[0]
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    filtered_lines = [line for line in lines if line.strip() and line.strip() != '96']
    
    atomic_data = []
    for line in filtered_lines:
        parts = line.split()
        if len(parts) == 7:
            species = parts[0]
            x, y, z = map(float, parts[1:4])
            fx, fy, fz = map(float, parts[4:7])
            atomic_data.append([species, x, y, z, fx, fy, fz])
    
    columns = ["Species", "X", "Y", "Z", "Fx", "Fy", "Fz"]
    df_atoms = pd.DataFrame(atomic_data, columns=columns)
    
    energy_values = []
    for line in lines:
        if "energy=" in line:
            energy_str = line.split("energy=")[1].split()[0]
            try:
                energy_value = float(energy_str)
                energy_values.append(energy_value)
            except ValueError:
                continue
    if job_type == 'Train':
        data_length = 1500
    if job_type == 'Test':
        data_length = 3000
    
    energy_values = (energy_values + [0.0] * (data_length - len(energy_values)))[:data_length]
    
    atom_number_dict = {'Hf':72, 'O':8}
    
    num_atoms = 96  # 주어진 데이터에 따른 원자의 수
    total_atoms = len(df_atoms)
    batch_size = total_atoms // num_atoms  # 배치 크기 계산
    species = df_atoms['Species'].map(atom_number_dict).values.reshape(batch_size, num_atoms)
    x = df_atoms['X'].values.reshape(batch_size, num_atoms)
    y = df_atoms['Y'].values.reshape(batch_size, num_atoms)
    z = df_atoms['Z'].values.reshape(batch_size, num_atoms)
    fx = df_atoms['Fx'].values.reshape(batch_size, num_atoms)
    fy = df_atoms['Fy'].values.reshape(batch_size, num_atoms)
    fz = df_atoms['Fz'].values.reshape(batch_size, num_atoms)
    
    force_features, energy_features, atomic_energy_features  =[], [], []
    for idx in range(len(species)):
        coordinates = torch.tensor(np.stack((x[idx,:], y[idx,:], z[idx,:]), axis=-1), dtype=torch.float32)
        
        force_features.append(calculate_forces_energy(coordinates, species[0])[0])
        energy_features.append(calculate_forces_energy(coordinates,species[0])[1])
        atomic_energy_features.append(calculate_forces_energy(coordinates,species[0])[2])
            
    force_features          = torch.stack(force_features) 
    energy_features         = torch.stack(energy_features) 
    atomic_energy_features  = torch.stack(atomic_energy_features)
    
    atomic_data_np = {
                        'atoms': species,
                        'x': x,
                        'y': y,
                        'z': z,
                        'fx': fx,
                        'fy': fy,
                        'fz': fz,
                        'energy': np.array(energy_values).reshape(batch_size, -1),
                        'energy_uncertainty': np.zeros((batch_size, num_atoms)),
                        'force_features'    : np.array(force_features),
                        'energy_features'   : np.array(energy_features),
                        'atomic_energy_features'   : np.array(atomic_energy_features) 
                    }
    
    with h5py.File('./data/' + job_type + '_db.h5' ,'w') as h5_file:
        for key, value in atomic_data_np.items():
            h5_file.create_dataset(key, data=value)



def load_h5(filename):
    data = {}
    with h5py.File(filename, 'r') as h5_file:
        data['atoms']          = h5_file['atoms'][:]
        data['x']              = h5_file['x'][:]
        data['y']              = h5_file['y'][:]
        data['z']              = h5_file['z'][:]
        data['fx']             = h5_file['fx'][:]
        data['fy']             = h5_file['fy'][:]
        data['fz']             = h5_file['fz'][:]
        data['energy']         = h5_file['energy'][:]
        data['force_features'] = h5_file['force_features'][:]
        data['energy_features']= h5_file['energy_features'][:]
        data['atomic_features']= h5_file['atomic_energy_features'][:]
    
    return data



 
