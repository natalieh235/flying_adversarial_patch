import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path

import numpy as np

import yaml

import torch
from util import load_dataset
from patch_placement import place_patch
from attacks import get_transformation

from collections import defaultdict

def img_placed_patch(targets, patch, scale_norm, tx_norm, ty_norm, img_idx=0):
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'
    train_set = load_dataset(path=dataset_path, batch_size=1, shuffle=True, drop_last=False, num_workers=0)

    base_img, ground_truth = train_set.dataset.__getitem__(img_idx)
    base_img = base_img.unsqueeze(0) / 255.

    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0) 

    scale_norm = torch.tensor(scale_norm)
    tx_norm = torch.tensor(tx_norm)
    ty_norm = torch.tensor(ty_norm)


    final_images = []
    for target_idx in range(len(targets)):
        transformation_matrix = get_transformation(scale_norm[target_idx], tx_norm[target_idx], ty_norm[target_idx])
        
        final_images.append(place_patch(base_img, patch, transformation_matrix).numpy()[0][0])

    return np.array(final_images)


def plot_results(path):

    path = Path(path)
    with open(path / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T

    mode = settings['mode']

    # print(targets)

    optimization_patches = np.load(path / 'patches.npy')
    # print(optimization_patches.shape)
    
    optimization_patch_losses = np.load(path / 'patch_losses.npy')
    # print(optimization_patch_losses.shape)

    optimization_pos_losses = np.load(path / 'position_losses.npy')
    # print(optimization_pos_losses.shape)

    all_sf, all_tx, all_ty = np.load(path / 'positions_norm.npy')
    # print(all_sf.shape)
    # print(all_tx.shape)
    # print(all_ty.shape)
    
    train_losses = np.load(path / 'losses_train.npy')
    test_losses = np.load(path / 'losses_test.npy')

    # print(train_losses.shape)
    # print(test_losses.shape)

    boxplot_data = np.load(path / 'boxplot_data.npy')
    boxplot_data = np.rollaxis(boxplot_data, 2, 1)
    # print(boxplot_data.shape)

    img_w_patch = img_placed_patch(targets, optimization_patches[-1], scale_norm=all_sf[-1], tx_norm=all_tx[-1], ty_norm=all_ty[-1])
    #print(img_w_patch.shape)

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

        if mode == 'hybrid' or mode == 'split':
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'Loss position optimization for all iterations')
            for target_idx, target in enumerate(targets):
                ax.plot(optimization_pos_losses[target_idx], label=f'target {target}')
            ax.set_xlabel('iteration')
            ax.set_ylabel('MSE')
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
            ax.set_ylabel('MSE')
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

        for target_idx, target in enumerate(targets):
            fig, ax = plt.subplots(1,1)
            ax.imshow(img_w_patch[target_idx], cmap='gray')
            ax.set_title(f'Placed patch after optimization, target {target}')
            pdf.savefig(fig)
            plt.close()

def combine_arrays(paths, name):
    arr = []
    for path in paths:
        file = path / name
        # print(file)
        if file.exists():
            arr.append(np.load(file))
    return np.array(arr)

def gen_dict(paths):
    gen_dict = {
        'patches': combine_arrays(paths, 'patches.npy'),
        'patch_loss': combine_arrays(paths, 'patch_losses.npy'),
        'pos_loss': combine_arrays(paths, 'position_losses.npy'),
        'positions': combine_arrays(paths, 'positions_norm.npy'),
        'train_loss': np.rollaxis(combine_arrays(paths, 'losses_train.npy'), 2, 0),
        'test_loss': np.rollaxis(combine_arrays(paths, 'losses_test.npy'), 2, 0),
        'boxplot_data': np.rollaxis(combine_arrays(paths, 'boxplot_data.npy'), 1, 0)
    }
    return gen_dict

def plot_single(data, title, xlabel='iterations', ylabel='MSE', mean=True, legend=True):
    fig, ax = plt.subplots(1, 1)
    ax.plot(data, color='lightsteelblue')
    if mean:
        ax.plot(np.mean(data, axis=1), label='mean', color='darkorange')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()
    return fig

def plot_multi(multi_data, title, labels, colors, xlabel='iterations', ylabel='MSE', mean=True, legend=True):
    fig, ax = plt.subplots(1, 1)
    for data, label, color in zip(multi_data, labels, colors):
        ax.plot(data, color=color)
        if mean:
            ax.plot(np.mean(data, axis=1), label='mean '+ label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend()

    return fig

def show_multi_placed(result, targets, mode):
    final_images = []
    for run_idx, position in enumerate(result['positions']):
        scale_norm, tx_norm, ty_norm = position[:, -1, :]
        last_patch = result['patches'][run_idx][-1]
        final_images.append(img_placed_patch(targets, last_patch, scale_norm, tx_norm, ty_norm))
    final_images = np.rollaxis(np.array(final_images), 1, 0)
    
    best_idx = np.argmin(np.mean(result['test_loss'][:, :, -1], axis=0))
    print(best_idx)

    # print(final_images.shape)
    # print(final_images[0][0].shape)
    #figures = []
    #for target_idx, target in enumerate(targets):
    fig, ax = plt.subplots(2, 1)
    #fig.suptitle(f"Patches at optimal positions, mode: {mode}, target {target}")
    img_idx = 0
    for row in range(2):
        #for column in range(5):
        ax[row].imshow(final_images[row, best_idx], cmap='gray')
        ax[row].set_title(f'target prediction: x={targets[row][0]}, y={targets[row][1]}, z={targets[row][2]}')
        ax[row].axis('off')
            #if img_idx == best_idx[target_idx]:
                #ax[row, column].set_title(f"run {img_idx}, best")
            #else:
                #ax[row, column].set_title(f"run {img_idx}")
            #img_idx +=1
        #figures.append(fig)
    return fig


def gen_boxplots(data, title, labels, ylabel='MSE'):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.boxplot(data, 1, 'D', labels=labels)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    return fig

def eval_multi_run(path, modes=['fixed', 'joint', 'split', 'hybrid']):
    path = Path(path)

    results = defaultdict(list)
    for mode in modes:
        paths = sorted([p for p in path.glob(f"{mode}*")])
        results[mode] = gen_dict(paths)

    with open(paths[0] / 'settings.yaml') as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)
        
    runs = len(paths)

    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T

    for mode in modes:
        all = results[mode]['test_loss'][:, :, -1]
        print(all.shape)
        print(mode, "mean", np.mean(all), "std", np.std(all))

    with PdfPages(Path(path) / 'combined_result.pdf') as pdf:
        
        boxplot_means = [] 

        for mode in modes:
            boxplot_mean = []

            patch_losses = plot_single(results[mode]['patch_loss'].T, title=f'Patch loss, mode: {mode}, runs: {runs}')
            pdf.savefig(patch_losses)
            plt.close()

            for i, target in enumerate(targets):
                
                if mode == 'split' or mode == 'hybrid':
                    pos_losses = plot_single(results[mode]['pos_loss'].T[i], title=f'Position loss, mode: {mode}, runs: {runs}, target: {target}')
                    pdf.savefig(pos_losses)
                    plt.close()


                train_test_loss = plot_multi([results[mode]['train_loss'][i].T, results[mode]['test_loss'][i].T], title=f"Evaluation loss after each iteration, mode: {mode}, runs: {runs}, target: {target}", labels=['train loss', 'test loss'], colors=['lightsteelblue', 'peachpuff'])
                pdf.savefig(train_test_loss)
                plt.close()
                
                boxplot_mean.append(np.mean(results[mode]['boxplot_data'][i], axis=0))
        
            boxplot_means.append(boxplot_mean)

            # --- place patches at optimal position 
            fig = show_multi_placed(results[mode], targets, mode)
            #for fig in figures:
            pdf.savefig(fig)
            plt.close()  
        
        # --- create box plots
        boxplot_means = np.rollaxis(np.array(boxplot_means), 1, 0)
        print(boxplot_means.shape)
        for target, means in zip(targets, boxplot_means):
            base_img_mean = np.mean(means[:, 0, :], axis=0)
            base_patch_mean = np.mean(means[:, 1, :], axis=0)

            #boxplot = gen_boxplots([base_img_mean, base_patch_mean, *means[:, 2]], title=f'MSE for each test image, taregt: {target}, runs {runs}', labels=['base images', 'starting patch', *modes], ylabel='mean MSE')
            boxplots_base_start_fixed = gen_boxplots([base_img_mean, base_patch_mean, means[0, 2]], title=f'target: x = {target[0]}, y = {target[1]}, z={target[2]}', labels=['base images', 'starting patch', modes[0]], ylabel='Test loss [m]')
            boxplots_joint_split_hybrid = gen_boxplots([*means[1:, 2]], title=f'target: x = {target[0]}, y = {target[1]}, z={target[2]}', labels=[*modes[1:]], ylabel='Test loss [m]')
            pdf.savefig(boxplots_base_start_fixed)
            pdf.savefig(boxplots_joint_split_hybrid)
            pdf.close


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--final', action='store_true')
    parser.add_argument('--path', default='eval/')
    parser.add_argument('--modes', nargs='+', default=['fixed', 'joint', 'split', 'hybrid'])
    args = parser.parse_args()

    if args.final:
        eval_multi_run(args.path, args.modes)
    else:
        plot_results(args.path)