from cmath import acos
from tkinter import Y
from matplotlib.pyplot import axis
from sympy import Float
from torch.utils.data import Dataset
import os
import torch
import numpy as np
import random
import math
import matplotlib.pylab as plt
from sklearn import preprocessing

def Angle(a):
    unit_x = [1,0,0]
    if a[1] >=0:
         return np.arccos(np.dot(a, unit_x)/ (np.linalg.norm(a) * np.linalg.norm(unit_x)))
    else:
        return 2*np.pi - np.arccos(np.dot(a, unit_x)/ (np.linalg.norm(a) * np.linalg.norm(unit_x)))


class GridDataset(Dataset):
    def __init__(self, root_dir="data", split="train", force_scaler=None, coords_scaler=None):
        self.data_path = f"{root_dir}/{split}" 
        self.data = [folder for folder in os.listdir(self.data_path)]
        self.split = split
        self.force_scaler = force_scaler
        self.coords_scaler = coords_scaler

    def __len__(self):
        return len(self.data)
        #return 1

    def __getitem__(self, idx):
        folder_name = self.data[idx]
        full_path = f"{self.data_path}/{folder_name}"

        forces = []
        angles = []
        with open(f'{full_path}/Input.txt','r') as f:
            for i, line in enumerate(f):
                if i < 5:
                    x, y, z = line.rstrip('\n').split(',')
                    vector = [float(x),float(y), float(z)]
                    angles.append(Angle(vector))
                else:
                    F = line.rstrip('\n')
                    forces.append(float(F))

        
        

        if self.force_scaler != None:
            forces = self.force_scaler.transform(np.array(forces).reshape(-1,3))
            forces = torch.from_numpy(np.array(forces)).float().squeeze(0)
        else:
            forces = torch.tensor(np.array(forces))

        FEM_stress = []
        FEM_disp = []
        coords = []
        coords_original = []
        index_list = []
        sum_list = []

        angle_list = []



        for i in range(5):
            angle = angles[i]
            rot_mat = np.array([[np.cos(2*np.pi-angle), -np.sin(2*np.pi-angle), 0],[np.sin(2*np.pi-angle), np.cos(2*np.pi-angle), 0],[0, 0, 1]])

            angle_list.append(angle)
            with open(f'{full_path}/Stress_Box_{i}.txt','r') as f:
                for line in f:
                    ux, uy, uz, stress, x, y, z = line.rstrip('\n').split(',')
                    pt_coord_original = [float(x), float(y), float(z)]
                    pt_coord = pt_coord_original @ rot_mat.T

                    c_sum = float(x), float(y), float(z)
                    if c_sum not in sum_list:
                        pt_stress = float(stress)
                        pt_disp = [float(ux), float(uy), float(uz)]

                        FEM_stress.append(pt_stress)
                        coords.append(pt_coord)
                        coords_original.append(pt_coord_original)
                        FEM_disp.append(pt_disp)    
                        
                        index_list.append(int(i))
                    sum_list.append(c_sum)
 
        
        coords_original = torch.tensor(np.array(coords_original))
        index_list = torch.tensor(np.array(index_list))
        FEM_stress = torch.tensor(np.array(FEM_stress))
        FEM_disp = torch.tensor(np.array(FEM_disp))


        if self.coords_scaler != None:
            coords = self.coords_scaler.transform(np.array(coords).reshape(-1,3))
            coords = torch.from_numpy(np.array(coords)).float().squeeze(0)
        else:
            coords = torch.tensor(np.array(coords))
        

        return forces, coords, coords_original, FEM_stress, FEM_disp, index_list, angle_list

def main():

    dataset = GridDataset(split='train')
    forces, coords, coords_original, FEM_stress, FEM_disp, index_list, angle_list = dataset[0]

    

    coords_original = coords_original.cpu().squeeze().detach().numpy()
    index_list = index_list.cpu().squeeze().detach().numpy()
    FEM_disp = FEM_disp.cpu().squeeze().detach().numpy()*10
    max, min = coords_original.max().item(), coords_original.min().item()


    FEM_x = coords_original[:,0] + FEM_disp[:,0]
    FEM_y = coords_original[:,1] + FEM_disp[:,1]
    FEM_z = coords_original[:,2] + FEM_disp[:,2]

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(111, projection = '3d')
    ax.scatter(FEM_x,FEM_y,FEM_z, s=50, c = FEM_stress, cmap='viridis')

    ax.set_xlim(min,max)
    ax.set_ylim(min,max)
    ax.set_zlim(min,max)
    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()