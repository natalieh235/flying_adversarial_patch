import yaml
import subprocess 
import numpy as np

from pathlib import Path

import os

import shlex



if __name__=="__main__":

    with open('dataset.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
    
    for i in range(10):
        patch_size = settings['patch']['size']
        path = Path(f'results/dataset{patch_size[0]}x{patch_size[1]}/{i}/')

        # get target values in correct shape and move tensor to device
        targets = [values for _, values in settings['targets'].items()]
        targets = np.array(targets, dtype=float).T
        # targets = torch.from_numpy(targets).to(device).float()

        number_targets = np.random.randint(1, 3)

        # generate random amount of targets in reasonable ranges
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