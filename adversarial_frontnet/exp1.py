import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp


class Experiment1:
    def create_settings(self, base_settings, trials):
        all_settings = []
        for i in range(trials):
            for mode in ['fixed', 'split', 'joint', 'hybrid']:
                s = copy.copy(base_settings)
                s['mode'] = mode
                s['path'] = "eval/exp1a/" + mode + str(i)
                all_settings.append(s)
        return all_settings

    def stats(self, all_settings):
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

def main():
    e = Experiment1()
    exp.exp(e)


if __name__ == '__main__':
    main()
