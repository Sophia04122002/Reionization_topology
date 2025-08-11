from __future__ import print_function 
from scipy.ndimage import gaussian_filter1d
import tools21cm as t2c
import os, sys 
from scipy.interpolate import interp1d
import math
from scipy.stats import gaussian_kde
import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib
matplotlib.rcParams.update({'font.size': 11})

ngrid= 50
simulation_cube = (ngrid, ngrid, ngrid)
ionized_box = np.zeros(simulation_cube, dtype=int)

xv, yv, zv = np.meshgrid(np.arange(ngrid), np.arange(ngrid), np.arange(ngrid), indexing='ij',sparse=True)

# helper functions 
def periodic_distance(coord1, coord2, box_size):
     delta = np.abs(coord1 - coord2)
     return np.minimum(delta, box_size - delta) #in periodic universe, if photons lies at 0th pos and moves to 9th pos. then min distance is 1 (9 -> 0), vs 9-0=9

def apply_periodic_index(idx, ngrid):
    return idx % ngrid

def apply_periodic_pos(pos, ngrid):
    return np.mod(pos, ngrid)

def choose_random_direction(): 
    theta = np.arccos(np.random.uniform(-1, 1))  #theta range: 0 to pi
    phi = np.random.uniform(0, 2 * np.pi)
    x_vec = np.sin(theta) * np.cos(phi)
    y_vec = np.sin(theta) * np.sin(phi)
    z_vec = np.cos(theta)
    return np.array([x_vec, y_vec, z_vec], dtype=np.float64)

def launching_rays(start_idx, random_direction_vector, ionized_mask, ngrid, step_size=1.0): 
    pos = np.array(start_idx, dtype=np.float64)
    distance=0.0
    max_distance = 3 * ngrid 
    while distance < max_distance:
        pos += random_direction_vector * step_size
        pos = apply_periodic_pos(pos, ngrid) 
        idx = np.floor(pos).astype(int) 
        idx = apply_periodic_index(idx, ngrid)
        if not ionized_mask[tuple(idx)]:
            return distance #* cell_size 
        distance += step_size      
    return max_distance #* cell_size  

# initialize box 
center_x = np.random.randint(0, ngrid)
center_y = np.random.randint(0, ngrid)
center_z = np.random.randint(0, ngrid)

dx = periodic_distance(xv, center_x, ngrid)
dy = periodic_distance(yv, center_y, ngrid)
dz = periodic_distance(zv, center_z, ngrid)

dist2 = dx**2 + dy**2 + dz**2
radius = 10 
mask = dist2 <= radius**2
ionized_box[mask] = 1
ionized_coords = np.argwhere(ionized_box == 1) #ger coordinates of ionized voxels


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x, y, z = ionized_coords[:, 0], ionized_coords[:, 1], ionized_coords[:, 2]
ax.scatter(x, y, z, c='purple', marker='o', s=10, label='Ionized Bubble')
ax.set_xlabel('x', fontsize=16)
ax.set_ylabel('y', fontsize=16)
ax.set_zlabel('z', fontsize=16)
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])
ax.legend(fontsize=14, frameon=False)
plt.tight_layout()
plt.show()

#2D visualization along z
plt.imshow(ionized_box[:, :, center_z], cmap='viridis')
plt.title(f"Sliced plot at z = {center_z} with R = {radius}", fontsize=14)
cbar = plt.colorbar(label='Ionized Fraction')
cbar.ax.tick_params(labelsize=17)        
cbar.set_label('Ionized Fraction', fontsize=17) 
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


#mfp
mean_free_paths = []
num_iterations = 1000000# 10000

threshold = 0.5
for _ in range(num_iterations):
    ionized_mask= (ionized_box == 1) # it's like my threshold
    idx = np.random.randint(0, ngrid, size=3)
    if not ionized_mask[tuple(idx)]:
        continue
    vector_direction = choose_random_direction()
    mfp = launching_rays(idx, vector_direction, ionized_mask, ngrid)
    mean_free_paths.append(mfp)


#histogram mfps
mfp_array = np.array(mean_free_paths)
print("Mean Free Paths:", mfp_array)
plt.hist(mfp_array, bins=ngrid, density=True)
plt.xlabel('R (cell units)', fontsize=16)
plt.ylabel('PDF', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.show()


smooth_R = np.linspace(1, ngrid, 64)
R_dense = np.linspace(smooth_R[0], smooth_R[-1], 50)

kde = gaussian_kde(mean_free_paths)
smooth_pdf = kde(smooth_R)
bsd = smooth_R * smooth_pdf
bsd /= np.trapz(bsd, smooth_R)
print("bsd", np.trapz(bsd, smooth_R))
curve_bsd = interp1d(smooth_R, bsd, kind='cubic', fill_value="extrapolate")
dense_bsd = curve_bsd(R_dense)
# peak
peak_idx = np.argmax(dense_bsd) 
R_peak = R_dense[peak_idx]
peak = dense_bsd[peak_idx]




r_mfp, dn_mfp = t2c.mfp(ionized_mask, boxsize=ngrid, iterations=1000000)
tools_mask = r_mfp**2 <= radius**2
r_mfp_dense = np.linspace(r_mfp[0], r_mfp[-1], 50)
#dn_mfp = dn_mfp[tools_mask]
dn_mfp /= np.trapz(dn_mfp, r_mfp)  
print("area pdf tools21", np.trapz(dn_mfp, r_mfp))
curve_dn = interp1d(r_mfp, dn_mfp, kind='cubic', fill_value="extrapolate")
dense_dn = curve_dn(r_mfp_dense)



plt.figure(figsize=(12,6))
plt.plot(R_dense, dense_bsd, label='This work', color='black') 
plt.plot(r_mfp, dense_dn, linestyle='--',label = 'Literature', color='orange')
plt.vlines(x=radius, ymin=0, ymax = peak,linestyle='-.', color='red', label='Real Bubble Size')
#plt.axvline(x=R_peak, linestyle='--', color='blue', label='Peak of BSD')
plt.legend(fontsize=12, loc='upper right', frameon=False)
plt.xscale('log')
plt.xlabel(r'R (voxel units)', fontsize=16)
plt.ylabel(r"$R\,dP/dR$", fontsize=16)
plt.tick_params(labelsize=14)
plt.show()


