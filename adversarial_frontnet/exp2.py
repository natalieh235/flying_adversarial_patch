import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp


class Experiment2:
    def create_settings(self, base_settings, trials):
        all_settings = []
        for i in range(trials):
            for j in range(1, len(base_settings['targets']['x'])+1):
                s = copy.deepcopy(base_settings)
                s['targets']['x'] = base_settings['targets']['x'][0:j]
                s['targets']['y'] = base_settings['targets']['y'][0:j]
                s['targets']['z'] = base_settings['targets']['z'][0:j]
                s['path'] = "eval/exp2/" + str(j) + "_" + str(i)
                all_settings.append(s)
        return all_settings

    def stats(self, all_settings):
        # output statistics
        result = defaultdict(list)
        for settings in all_settings:
            p = Path(settings['path'])
            test_losses = np.load(p / 'losses_test.npy')
            print(len(settings['targets']['x']), np.mean(test_losses[-1]))
            result[len(settings['targets']['x'])].append(test_losses[-1])

        for k, v in result.items():
            all = np.stack(v)
            print(k, "mean", np.mean(all), "std", np.std(all))

def main():
    e = Experiment2()
    exp.exp(e)


if __name__ == '__main__':
    main()