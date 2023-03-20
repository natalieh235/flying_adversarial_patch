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
from collections import defaultdict

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
    parser.add_argument('--file', default='exp1.yaml')
    parser.add_argument('--norun', action='store_true')
    parser.add_argument('-j', type=int, default=4)
    parser.add_argument('--trials', type=int, default=10)
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, on_sigterm)

    # SETTINGS
    with open(args.file) as f:
        base_settings = yaml.load(f, Loader=yaml.FullLoader)

    # create possible settings
    all_settings = []
    for i in range(args.trials):
        for mode in ['fixed', 'split', 'joint', 'hybrid']:
            s = copy.copy(base_settings)
            s['mode'] = mode
            s['path'] = "eval/exp1a/" + mode + str(i)
            all_settings.append(s)

    if not args.norun:
        # start 4 worker processes
        with Pool(processes=args.j) as pool:
            for i in pool.imap_unordered(run_attack, all_settings):
                pass

    # output statistics
    result = defaultdict(list)
    for settings in all_settings:
        p = Path(settings['path'])
        test_losses = np.load(p / 'losses_test.npy')
        # print(settings['mode'], test_losses[-1])
        result[settings['mode']].append(test_losses[-1])

    for k, v in result.items():
        all = np.stack(v)
        print(k, "mean", np.mean(all), "std", np.std(all))


if __name__ == '__main__':
    main()