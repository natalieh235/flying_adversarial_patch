import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

import numpy as np

import yaml

def plot_results(path):
    # TODO: read targets from config!

    path = Path(path)
    with open('settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T

    mode = settings['mode']

    print(targets)

    optimization_patches = np.load(path / 'patches.npy')
    print(optimization_patches.shape)
    
    optimization_patch_losses = np.load(path / 'patch_losses.npy')
    print(optimization_patch_losses.shape)

    optimization_pos_losses = np.load(path / 'position_losses.npy')
    print(optimization_pos_losses.shape)

    all_sf, all_tx, all_ty = np.load(path / 'positions_norm.npy')
    print(all_sf.shape)
    print(all_tx.shape)
    print(all_ty.shape)
    
    train_losses = np.load(path / 'losses_train.npy')
    test_losses = np.load(path / 'losses_test.npy')

    print(train_losses.shape)
    print(test_losses.shape)

    boxplot_data = np.load(path / 'boxplot_data.npy')
    boxplot_data = np.rollaxis(boxplot_data, 2, 1)
    print(boxplot_data.shape)


    with PdfPages(Path(path) / 'result.pdf') as pdf:
        for idx, patch in enumerate(optimization_patches):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'Patch at training iteration {idx}')
            ax.imshow(patch, cmap='gray')
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Loss for {mode} patch optimization')
        ax.plot(optimization_patch_losses.flatten())
        ax.set_xlabel('iteration')
        ax.set_ylabel('MSE')
        pdf.savefig(fig)
        plt.close(fig)

        if mode != "joint":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'Loss position optimization for all iterations')
            for target_idx, target in enumerate(targets):
                ax.plot(optimization_pos_losses[target_idx], label=f'target {target}')
            ax.set_xlabel('iteration')
            ax.set_ylabel('mean l2 distance')
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

        for target_idx, target in enumerate(targets):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'Losses after each patch & position optimization')
            ax.plot(train_losses[..., target_idx], label=f'train set, target {target}')
            ax.plot(test_losses[..., target_idx], label=f'test set, target {target}')
            ax.legend()
            ax.set_xlabel('iteration')
            ax.set_ylabel('mean l2 distance')
            pdf.savefig(fig)
            plt.close(fig)


        # for idx in range(len(optimization_pos_losses)):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f'Loss patch optimization, iteration {idx}')
        #     ax.plot(optimization_pos_losses[idx].cpu())
        #     ax.set_xlabel('training steps')
        #     ax.set_ylabel('mean l2 distance')
        #     pdf.savefig(fig)
        #     plt.close(fig)


        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'scale factor for all iterations')
        for target_idx, target in enumerate(targets):
            ax.plot(all_sf[:, target_idx], label=f'target {target}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('scale factor')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        # for idx in range(all_sf.shape[0]):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f'scale factor, iteration {idx}')
        #     ax.plot(all_sf[idx].cpu())
        #     ax.set_xlabel('training steps')
        #     ax.set_ylabel('scale factor')
        #     pdf.savefig(fig)
        #     plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'tx for all iterations')
        for target_idx, target in enumerate(targets):
            ax.plot(all_tx[:, target_idx], label=f'target {target}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('tx')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'ty for all iterations')
        for target_idx, target in enumerate(targets):
            ax.plot(all_ty[:, target_idx], label=f'target {target}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('ty')
        ax.legend()
        pdf.savefig(fig)
        plt.close(fig)

        for target_idx, target in enumerate(targets):
            fig, ax = plt.subplots(1, 1)
            ax.boxplot(boxplot_data[target_idx], 1, 'D', labels=['base images', 'starting patch', 'optimized patch'])
            ax.set_title(f'boxplots, patches placed at optimal position for target {target}')
            ax.set_ylabel('MSE(target, prediction)')
            # axs[0].set_title('base images')
            # axs[1].boxplot(rel_y, 1, 'D')
            # axs[1].set_title('y - target y')
            pdf.savefig(fig)
            plt.close(fig)



if __name__=="__main__":
    import sys

    path = sys.argv[1]
    plot_results(path)




### depracated
# for idx in range(len(all_ty)):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_title(f'ty, iteration {idx}')
#     ax.plot(all_ty[idx].cpu())
#     ax.set_xlabel('training steps')
#     ax.set_ylabel('ty')
#     pdf.savefig(fig)
#     plt.close(fig)

# for target_idx, target in enumerate(targets):
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     ax.set_title(f'Placed patch after optimization, target {target.cpu().item()}')
#     ax.imshow(final_images[target_idx][0][0].detach().cpu().numpy(), cmap='gray')
#     plt.axis('off')
#     pdf.savefig(fig)
#     plt.close(fig)

# fig = plot_saliency(base_img, ground_truth, model)
# fig.suptitle(f'y = {prediction[1].detach().cpu().item()}')
# pdf.savefig(fig)
# plt.close(fig)

# fig = plot_saliency(mod_start, ground_truth, model)
# fig.suptitle(f'y = {prediction_start[1].detach().cpu().item()}')
# pdf.savefig(fig)
# plt.close(fig)

# fig = plot_saliency(mod_img, ground_truth, model)
# fig.suptitle(f'y = {prediction_mod[1].detach().cpu().item()}')
# pdf.savefig(fig)
# plt.close(fig)