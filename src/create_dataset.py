import yaml
import subprocess 
import numpy as np
from pathlib import Path
import os
import shlex
import pickle
import argparse

def train(idx_start=0, idx_end=100):
    with open('dataset.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    
    for i in range(idx_start, idx_end):
        patch_size = settings['patch']['size']
        path = Path(f'results/dataset{patch_size[0]}x{patch_size[1]}/{i}/')

        targets = [values for _, values in settings['targets'].items()]
        targets = np.array(targets, dtype=float).T

        number_targets = np.random.randint(1, 4)

        random_target_x = np.random.uniform(0,2,number_targets)
        random_target_y = np.random.uniform(-1,1,number_targets,)
        random_target_z = np.random.uniform(-0.5,0.5,number_targets,)

        # overwrite settings
        settings['path'] = str(path)
        settings['targets']['x'] =  random_target_x.tolist()
        settings['targets']['y'] = random_target_y.tolist()
        settings['targets']['z'] = random_target_z.tolist()
        settings['patch']['size'] = patch_size

        os.makedirs(path, exist_ok = True)
        with open(path / 'settings.yaml', 'w') as f:
            yaml.dump(settings, f)

        command = shlex.split(f"sbatch dataset.sh {str(path / "settings.yaml")}")
        subprocess.run(command)

def save_pickle(idx_start=0, idx_end=100):
    with open('dataset.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    
    patch_size = settings['patch']['size']
    path = Path(f'results/dataset{patch_size[0]}x{patch_size[1]}/')

    file_paths = list(path.glob('[0-9]*/patches.npy'))
    file_paths.sort(key=lambda path: int(path.parent.name))
    
    patches = np.array([np.load(file_path)[-1][0][0] for file_path in file_paths[idx_start:idx_end]])
    print(patches.shape)
    with open('80x80patches.pickle', 'wb') as f:
        pickle.dump(patches, f, pickle.HIGHEST_PROTOCOL)


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['train', 'save'])
    parser.add_argument('idx', type=int, metavar='N', nargs='+', default=[100])
    args = parser.parse_args()


    if len(args.idx) < 2:
        idx_start = 0
        idx_end = args.idx[0]
    else:
        idx_start, idx_end = np.sort(args.idx)

    if args.mode == 'train':
        train(idx_start, idx_end)
    
    if args.mode == 'save':
        save_pickle(idx_start, idx_end)