import yaml
from multiprocessing import Pool
import argparse
import copy
import tempfile
from pathlib import Path
import subprocess
import signal
import os
import numpy as np

def run_attack(settings):
    with tempfile.TemporaryDirectory() as tmpdirname:
        filename = Path(tmpdirname) / 'settings.yaml'
        with open(filename, 'w') as f:
            yaml.dump(settings, f)

        subprocess.run(["python3", "adversarial_frontnet/attacks.py", "--file", filename])

def on_sigterm():
    # kill the whole process group
    os.killpg(os.getpid(), signal.SIGTERM)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='settings.yaml')
    parser.add_argument('--norun', action='store_true')
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, on_sigterm)

    # SETTINGS
    with open(args.file) as f:
        base_settings = yaml.load(f, Loader=yaml.FullLoader)

    # create possible settings
    all_settings = []
    for mode in ['fixed', 'split', 'joint', 'hybrid']:
        s = copy.copy(base_settings)
        s['mode'] = mode
        s['path'] = "eval/" + mode + "0"
        all_settings.append(s)

    if not args.norun:
        # start 4 worker processes
        with Pool(processes=4) as pool:
            for i in pool.imap_unordered(run_attack, all_settings):
                pass

    # output statistics
    for settings in all_settings:
        p = Path(settings['path'])
        test_losses = np.load(p / 'losses_test.npy')
        print(settings['mode'], test_losses[-1])


if __name__ == '__main__':
    main()