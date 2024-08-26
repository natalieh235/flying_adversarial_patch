import yaml
import subprocess 
import numpy as np
from pathlib import Path
import os
import shlex
import pickle
import argparse

import torch
from patch_placement import place_patch
from util import load_model, load_dataset

from yolo_bounding import YOLOBox


def gen_T(coeffs):
    T = np.zeros((2,3))
    T[0, 0] = T[1, 1] = coeffs[0] # sf
    T[0, 2] = coeffs[1] # tx
    T[1, 2] = coeffs[2] # ty
    # T[2, 2] = 1.

    return torch.tensor(T, dtype=torch.float32)

def load_targets(patch_path):
    with open(patch_path) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).float()
    return targets

def get_coeffs(patch_path):
    # parent_folder = patch_path.parent
    T_coeffs = np.load(patch_path) # shape: [sf, tx, ty], epochs, num_targets, 1, 1
    # # should shape: num_targets, [sf, tx, ty]
    T_coeffs = T_coeffs[:, :, 0, 0].T

    return T_coeffs

def get_Ts(T_coeffs):
    Ts = [gen_T(coeffs) for coeffs in T_coeffs]
    return Ts

def calc_loss(target, patch, T, img, model):
    mod_img_pt = place_patch(img.to(patch.device), patch, T.unsqueeze(0).to(patch.device), random_perspection=False) 
    x, y, z, yaw = model(mod_img_pt)
    predicted_pose = torch.hstack((x, y, z))[0].detach().cpu()
    # print(predicted_pose, target)
    mse = torch.nn.functional.mse_loss(target.detach().cpu(), predicted_pose).item()
    return mse

def idx_best_target(targets, patch, Ts, img, model):
    losses = np.array([calc_loss(target, patch, T, img, model) for target, T in zip(targets, Ts)])
    return np.argmin(losses)

def get_best_target_pos(patch, patch_path, dataset, model, device):
    targets = load_targets(patch_path).to(device)
    coeffs = get_coeffs(patch_path)
    Ts = get_Ts(coeffs)

    patch = torch.tensor(patch, device=device).float().unsqueeze(0).unsqueeze(0)

    best_targets = []
    for data in dataset:
        img, _ = data
        best_targets.append(idx_best_target(targets, patch, Ts, img, model))

    idx_best = np.argmax(np.bincount(np.array(best_targets)))

    return targets[idx_best].detach().cpu().numpy(), coeffs[idx_best]

def save_pickle(idx_start=0, idx_end=100):
    torch.cuda.empty_cache()
    print(torch.cuda.memory_allocated())
    path = 'results/yolo_patches'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_path = "pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle"
    dataset = load_dataset(dataset_path, batch_size=1, train=False, train_set_size=0.95)

    model = YOLOBox()

    patches = []
    targets = []
    coeffs = []

    print('MEMORY LEFT AFTER LOADING MODEL AND DATA', torch.cuda.memory_allocated())

    for i in range(idx_start, idx_end + 1):
        try:
            patch = np.load(f'{path}/last_patch_{i}.npy')[0,0,:,:] * 255.
            patches.append(patch)

            patch_targets = load_targets(f'{path}/settings_{i}.yaml').to(device)
            targets.append(patch_targets)

            patch_coeffs = get_coeffs(f'{path}/position_norm_{i}.npy')
            coeffs.append(patch_coeffs)
        except Exception as e:
            print(f'error loading patch {i}, {e}')

    # patches = np.array([np.load(f'{path}/last_patch_{i}.npy')[0,0,:,:] for i in range(idx_start, idx_end)]) * 255.
    patches = np.array(patches)
    print(patches.shape)
    all_images = dataset.dataset.data
    print(all_images.shape)


    print("lengths", len(patches), len(targets), len(coeffs))

    best_targets = []
    best_coeffs = []
    for i, (patch, p_targets, p_coeffs) in enumerate(zip(patches, targets, coeffs)):
        print('on patch ', i)
        losses = []
        patch = torch.tensor(patch, device=device).float().unsqueeze(0).unsqueeze(0)
        Ts = get_Ts(p_coeffs)
        for idx_target in range(len(p_targets)):
            T = Ts[idx_target]
            mod_img_pt = place_patch(all_images.to(patch.device), patch.repeat((len(all_images), 1, 1, 1)), T.unsqueeze(0).repeat(len(all_images), 1, 1).to(patch.device), random_perspection=False) 
            pred = model(mod_img_pt)
            pred = pred.detach().cpu()
            losses.append(torch.nn.functional.mse_loss(p_targets[idx_target].repeat(len(all_images), 1).detach().cpu(), pred).item())
        
        best_targets.append(p_targets[np.argmin(losses)].detach().cpu().numpy())
        best_coeffs.append(p_coeffs[np.argmin(losses)])

    targets = np.array(best_targets)
    positions = np.array(best_coeffs)

    with open('all_yolo_patches.pkl', 'wb') as f:
        pickle.dump([[patch/255., target, position] for patch, target, position in zip(patches, targets, positions)], f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    # save_pickle(1, 653)
    save_pickle(1, 653)