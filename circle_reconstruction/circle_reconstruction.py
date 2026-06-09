import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


#filename circles_numberOfCircles_ImageSize_noiseType_maxNumberOFCirclesInImage_MeanSizeMeanShiftXMeanShiftY
FILENAME = "circles_30k_64_wn_sc_106464"

#make circle map defined by radius and offsets
def make_circle_map(map_size: int = 32, radius: float = 10, cx: float = 16, cy: float = 16):
    x = np.linspace(0, map_size-1, map_size)
    y = np.linspace(0, map_size-1, map_size)
    X, Y = np.meshgrid(x, y)
    Z = (((X-cx)**2 + (Y-cy)**2) <= radius**2).astype(int)
    return Z

#make multiple (single) circle maps with random radii and offsets + documenting circle on each map
def make_random_circle_maps(n: int = 1, map_size: int = 32, radius_mean: float = 10, radius_var: float = 0, radius_min: float = 2, 
                            cx_mean: float = 16, cx_var: float = 0, cy_mean: float = 16, cy_var: float = 0, max_shift: float = 10, seed: int = 42):
    rng = np.random.default_rng(seed)
    radii = rng.normal(radius_mean, radius_var, n).clip(min = radius_min)
    cx_shifts = rng.normal(cx_mean, cx_var, n).clip(max=max_shift)
    cy_shifts = rng.normal(cy_mean, cy_var, n).clip(max=max_shift)

    circle_images = np.zeros((n, map_size, map_size))
    doc = []
    for i in range(n):
        circle_images[i, :, :] = make_circle_map(map_size, radii[i], cx_shifts[i], cy_shifts[i])
        doc.append({"index": i, "radius": round(radii[i], 3), "cx_shift": round(cx_shifts[i], 3), "cy_shift": round(cy_shifts[i], 3)})

    return circle_images, doc

#make map with multiple circles, each defined by radius and offset in list
def make_multiple_circle_map(map_size: int = 32, number_circles: int = 1,  radius: list[float] = [10], cx: list[float] = [16], cy: list[float] = [16]):
    x = np.linspace(0, map_size-1, map_size)
    y = np.linspace(0, map_size-1, map_size)
    X, Y = np.meshgrid(x, y)
    Z_list = []
    for i in range(number_circles):
        Z = (((X-cx[i])**2 + (Y-cy[i])**2) <= radius[i]**2).astype(int)
        Z_list.append(Z)
    Z_final = sum(Z_list)
    return Z_final

#make multiple maps with random number of circles given a max number of circles and mean circle parameters 
def make_random_multiple_circle_maps(n: int = 1, map_size: int = 32, max_number_circles: int = 1, radius_mean: float = 10, radius_var: float = 0, radius_min: float = 2, 
                            cx_mean: float = 16, cx_var: float = 0, cy_mean: float = 16, cy_var: float = 0, max_shift: float = 10, seed: int = 42):
    rng = np.random.default_rng(seed)

    circle_images = np.zeros((n, map_size, map_size))
    doc = []

    for i in range(n):
        number_circles = rng.integers(1, max_number_circles, endpoint=True)

        radii = rng.normal(radius_mean, radius_var, number_circles).clip(min=radius_min)
        cx_shifts = rng.normal(cx_mean, cx_var, number_circles).clip(max=max_shift)
        cy_shifts = rng.normal(cy_mean, cy_var, number_circles).clip(max=max_shift)

        circle_images[i, :, :] = make_multiple_circle_map(map_size, number_circles, radii, cx_shifts, cy_shifts)
        doc.append({"index": i, "radius": radii, "cx_shift": cx_shifts, "cy_shift": cy_shifts})

    return circle_images, doc

#make noise white noise map
def make_noise_maps(n: int = 1, map_size: int = 32, noise_max: float = 10, seed: int = 42):
    noise_maps = np.zeros((n, map_size, map_size))
    rng = np.random.default_rng(seed)
    for i in range(n):
        noise_maps[i, :, :] = rng.uniform(0, noise_max, size=(map_size, map_size))

    return noise_maps

#save maps in .npz file
def save_maps(maps, ground_truths, doc, file_name: str = "circle_maps"):
    radii = np.array([m["radius"] for m in doc])
    cx_shifts = np.array([m["cx_shift"] for m in doc])
    cy_shifts = np.array([m["cy_shift"] for m in doc])

    np.savez(file_name + ".npz", maps=maps, ground_truths=ground_truths, radii=radii, cx_shifts=cx_shifts, cy_shifts=cy_shifts)
    print("File saved as ", file_name, ".npz")
    return None

