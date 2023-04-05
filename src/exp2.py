import copy
from collections import defaultdict
from pathlib import Path
import numpy as np
import exp

# plotting
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class Experiment2:
    def create_settings(self, base_settings, trials, mode):
        all_settings = []
        for i in range(trials):
            for j in range(1, len(base_settings['targets']['x'])+1):
                s = copy.deepcopy(base_settings)
                s['targets']['x'] = base_settings['targets']['x'][0:j]
                s['targets']['y'] = base_settings['targets']['y'][0:j]
                s['targets']['z'] = base_settings['targets']['z'][0:j]
                s['path'] = "eval/exp2_d/" + str(j) + "_" + str(i)
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

        xs = []
        ys = []
        yerr = []
        for k, v in result.items():
            all = np.stack(v)
            print(k, "mean", np.mean(all), "std")
            xs.append(k)

            # loss_per_run = np.mean(all, axis=1)
            ys.append(np.mean(all))
            yerr.append(np.std(all))

        # change settings to match latex
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": "Helvetica",
            "font.size": 12,
            "figure.figsize": (5, 2),
        })

        with PdfPages(p.parent / 'exp2.pdf') as pdf:
            fig, ax = plt.subplots(constrained_layout=True)

            ax.plot(xs, ys)
            ax.fill_between(xs, np.asarray(ys)+np.asarray(yerr), np.asarray(ys)-np.asarray(yerr), alpha=0.3)

            ax.set_ylabel('Test Loss per Target [m]')
            ax.set_xlabel('Number of Targets per Patch')
            # ax.set_ylim(0,0.2)

            # ax.bar(xs, ys)
            # ax.errorbar(xs, ys, yerr)

            pdf.savefig(fig)
            plt.close()

def main():
    e = Experiment2()
    exp.exp(e)


if __name__ == '__main__':
    main()