import numpy as np
import yaml
import torch
import torch.multiprocessing as mp

import os
import argparse

from tqdm import trange

from pathlib import Path

from attacks import targeted_attack_joint
from util import load_dataset

def train_patch(settings, dataset_path, device, model, p_num):
    print("process :", p_num)

     # save settings directly to result folder for later use
    path = Path(settings['path'])

    lr_pos = settings['lr_pos']
    lr_patch = settings['lr_patch']
    num_hl_iter = settings['num_hl_iter']
    num_pos_restarts = settings['num_pos_restarts']
    num_pos_epochs = settings['num_pos_epochs']
    num_patch_epochs = settings['num_patch_epochs']
    batch_size = settings['batch_size']
    mode = settings['mode']
    num_patches = settings['num_patches']
    prob_weight = settings['prob_weight']
    scale_min = settings['scale_min']
    scale_max = settings['scale_max']

    stlc_target_offsets = torch.tensor([o['target'] for o in settings['stlc']['offsets']], dtype=torch.float, device=device)
    stlc_position_offsets = [o['position'] for o in settings['stlc']['offsets']]
    stlc_weights = [o['weight'] if 'weight' in o else 1.0 for o in settings['stlc']['offsets']]

    # get target values in correct shape and move tensor to device
    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).to(device).float()

    print(targets, targets.shape)

    # or start from a random patch
    if settings['patch']['mode'] == 'random':
        size = settings['patch']['size']
        patch_start = torch.rand(num_patches, 1, size[0], size[1]).to(device)

    print(patch_start.shape)

    # TODO: randomly flip to start from white or random?
    # # or start from a white patch
    # if settings['patch']['mode'] == 'white':
    #     size = settings['patch']['size']
    #     patch_start = torch.ones(num_patches, 1, size[0], size[1]).to(device)

    os.makedirs(path, exist_ok = True)
    with open(path / f'{size[0]}x{size[1]}' / p_num / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)

    # patch, loss_patch, positions, stats, stats_p = targeted_attack_joint(train_set, patch, model, optimization_pos_vectors[-1], A, targets=targets, lr=lr_patch, epochs=num_patch_epochs, path=path, prob_weight=prob_weight, scale_min=scale_min, scale_max=scale_max, target_offsets=stlc_target_offsets, position_offsets=stlc_position_offsets,

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='settings.yaml')
    parser.add_argument('--num_processes', type=int, default=1)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    # SETTINGS
    with open(args.file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    quantized = settings['quantized']
   
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    # model = load_model(path=model_path, device=device, config=model_config)
    # model.eval()
    print("Loading quantized network? ", quantized)
    if not quantized:
        # load full-precision network
        from util import load_model
        model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
        model_config = '160x32'
        model = load_model(path=model_path, device=device, config=model_config)
    else:
        # load quantized network
        from util import load_quantized
        model_path = 'misc/Frontnet.onnx'
        model = load_quantized(path=model_path, device=device)
    
    model.eval().share_memory()

    lock = mp.Lock()

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train_patch, args=(settings, dataset_path, device, model, rank,))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # num_patches = settings['num_patches']

    # # load the patch from misc folder
    # if settings['patch']['mode'] == 'face':
    #     patch_start = np.load(settings['patch']['path'])
    #     patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device) / 255.
    #     patch_start.clamp_(0., 1.)
    #     patch_start = torch.stack([patch_start[0].clone() for _ in range(num_patches)])

    # # or start from a random patch
    # if settings['patch']['mode'] == 'random':
    #     size = settings['patch']['size']
    #     patch_start = torch.rand(num_patches, 1, size[0], size[1]).to(device)

    # # or start from a white patch
    # if settings['patch']['mode'] == 'white':
    #     size = settings['patch']['size']
    #     patch_start = torch.ones(num_patches, 1, size[0], size[1]).to(device)

    # optimization_pos_losses = []
    # optimization_pos_vectors = []

    # optimization_patches = []
    # optimization_patch_losses = []

    # train_losses = []
    # test_losses = []

    # stats_all = []
    # stats_p_all = []

    # #positions = torch.FloatTensor(len(targets), num_patches, 3, 1).uniform_(-1., 1.).to(device)
    # sf = torch.FloatTensor(len(targets), num_patches, 1).uniform_(-1, 1.).to(device)
    # tx = torch.FloatTensor(len(targets), num_patches, 1).uniform_(-10, 80.).to(device)
    # ty = torch.FloatTensor(len(targets), num_patches, 1).uniform_(-10, 80.).to(device)
    # positions = torch.stack([sf, tx, ty]).moveaxis(0, 2)

    # optimization_pos_vectors.append(positions)

    # # num_patches x bitness x width x height
    # patch = patch_start.clone()
    # optimization_patches.append(patch.clone())

    # # assignment: we start by assigning all targets to all patches
    # A = np.ones((num_patches, len(targets)), dtype=np.bool_)
