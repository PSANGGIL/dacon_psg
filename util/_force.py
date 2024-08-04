import torch
import numpy as np

# Constants in atomic units (au)
k_e = 1.0  # Coulomb's constant in au (dimensionless in atomic units)
epsilon = 1.0e-21  # Depth of the potential well in J (au is typically used for energy)
sigma = 3.4e-10 / 0.529177  # Convert sigma from meters to au
e = 1.0  # Elementary charge in au
bohr_to_m = 0.529177  # Conversion factor from Bohr radius (au) to Ångström

# Function to compute Lennard-Jones force in atomic units
def lennard_jones_force(r, epsilon, sigma):
    r6 = (sigma / r)**6
    r12 = r6**2
    F = 24 * epsilon * (2 * r12 - r6) / r
    return F

# Function to compute Lennard-Jones potential in atomic units
def lennard_jones_potential(r, epsilon, sigma):
    r6 = (sigma / r)**6
    r12 = r6**2
    V = 4 * epsilon * (r12 - r6)
    return V

# Function to calculate forces and energy for a given batch in atomic units

def calculate_forces_energy(coordinates_batch, species_batch):
    num_atoms = coordinates_batch.shape[0]

    # Assign charges based on species (Hf: +4e, O: -2e)
    charges_batch = torch.tensor([4*e if sp == 'Hf' else -2*e for sp in species_batch], dtype=torch.float32)

    # Calculate pairwise distance vectors and magnitudes
    r_vec = coordinates_batch.unsqueeze(0) - coordinates_batch.unsqueeze(1)  # Shape (96, 96, 3)
    r_mag = torch.norm(r_vec, dim=2)  # Shape (96, 96)

    # Mask to avoid division by zero
    safe_r_mag = r_mag.clone()
    safe_r_mag[safe_r_mag == 0] = float('inf')  # Avoid division by zero

    # Calculate unit vectors (r_hat)
    r_hat = r_vec / safe_r_mag.unsqueeze(2)

    # Coulomb potential energy and force in atomic units
    q_i = charges_batch.unsqueeze(0)  # Shape (1, 96)
    q_j = charges_batch.unsqueeze(1)  # Shape (96, 1)
    V_coulomb = k_e * (q_i * q_j) / safe_r_mag  # Shape (96, 96)
    F_coulomb_mag = V_coulomb / safe_r_mag  # F = -dV/dr
    F_coulomb_vec = F_coulomb_mag.unsqueeze(2) * r_hat  # Shape (96, 96, 3)

    # Lennard-Jones potential energy and force in atomic units
    V_lj = lennard_jones_potential(safe_r_mag, epsilon, sigma)
    F_lj_mag = lennard_jones_force(safe_r_mag, epsilon, sigma)
    F_lj_vec = F_lj_mag.unsqueeze(2) * r_hat  # Shape (96, 96, 3)

    # Total force vector in atomic units
    F_total_vec = F_coulomb_vec + F_lj_vec  # Shape (96, 96, 3)

    # Set self-interaction forces and potential energy to zero
    F_total_vec[torch.eye(num_atoms, dtype=torch.bool)] = 0
    V_coulomb[torch.eye(num_atoms, dtype=torch.bool)] = 0
    V_lj[torch.eye(num_atoms, dtype=torch.bool)] = 0

    # Calculate the total force for each atom by summing over all interactions
    forces = F_total_vec.sum(dim=1)  # Shape (96, 3)

    # Calculate total potential energy
    total_energy = (V_coulomb + V_lj).sum() / 2  # Divided by 2 to account for double counting

    # Calculate the energy for each atom
    atomic_energies = (V_coulomb + V_lj).sum(dim=1)  # Sum of interaction energies for each atom

    return forces, total_energy, atomic_energies
