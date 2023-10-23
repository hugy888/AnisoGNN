# Aniso-GNN
 
## Setup
Install dependencies. Please check these links to install [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorchGeometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).
```bash
conda create --name Aniso python=3.9
pip install -r requirements.txt
```

## HDF5 Files
Grain IDs, euler angles for each grain and the mechanical property of RVEs are stored in Ni_raw_data.hdf5 and Al_raw_data.hdf5. 

First layer: 12 textures
  Uniaxial Compression (0/45/90 degrees rotations),
  Uniaxial Tension (0/45/90 degrees rotations),
  Plane Strain Compression (0/45/90 degrees rotations),
  Simple Shear (0/45/90 degrees rotations)

Second layer: 25 RVEs per texture and arrays storing E_modulus and yield_strength of RVEs; 

Third layer: Grain IDs and euler angles from each RVE. 

## Prepare Data
PyTorch datalists for graphs and RVE mechanical properties have been created as pickle files and saved in graph_data folder. The method to create those pickle files:
```bash
python write_data.py
```

## Run Model
The different cases presented in the original paper are:
  1. O-SAGE_Ni_E
  2. C-Aniso_Ni_E
  3. O-SAGE_Al_E
  4. C-Aniso_Al_E
  5. O-SAGE_Al_YS
  6. S-Aniso_Al_YS
  
For each case, four evaluations are performed and loss histories, parity plots, and model checkpoints are outputted.
Four evaluations:
  1. test_45_90_deg
  2. test_45_deg
  3. test_ran_30%
  4. test_90_deg

```
Usage: python model.py [OPTIONS]
Options:
  --cases INT                  Case number (e.g., 1 - 6, default: 1)
  --config INT                 Hyper-parameter configuration number (default: 0)
  --output_type STR            Type of output figures (png/svg, default: png)
```
