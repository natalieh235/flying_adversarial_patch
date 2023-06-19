import torch
import numpy as np
import argparse
import yaml
import os

from pathlib import Path

from tqdm import trange, tqdm

from util import gen_noisy_transformations, norm_transformation, get_transformation

from patch_placement import place_patch

from torch.nn.functional import mse_loss

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def multi_joint(dataset, patches, model, positions, targets, lr=3e-2, epochs=10, path="eval/"):

    patches_t = patches.clone().requires_grad_(True)
    positions_t = positions.clone().transpose(0, 1).requires_grad_(True)
    #print(positions_t.shape)

    opt = torch.optim.Adam([patches_t, positions_t], lr=lr)

    losses = []

    try:
        best_loss = np.inf
        best_stats = None
        best_stats_p = None

        for epoch in range(epochs):

            stats = np.zeros((len(patches_t), len(targets)))
            stats_p = np.zeros((len(patches_t), len(targets)))

            actual_loss = torch.tensor(0.).to(patches.device)
            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patches.device) / 255. # limit images to range [0-1]

                target_losses = []
                for k, (position, target) in enumerate(zip(positions_t, targets)):
                    # print(position.shape)
                    #old version
                    scale_factor, tx, ty = position[0]
                    noisy_transformations = gen_noisy_transformations(len(batch), scale_factor, tx, ty)
                    patch_batch = torch.cat([patches_t for _ in range(len(batch))])

                    mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations) 


                    # multi-version
                    # #scale_factor, tx, ty = position.transpose(1, 0)

                    # #print(scale_factor.shape)
                    
                    # # generate a transformation matrix of batch_size for each of the num_patches positions
                    # # this way, each patch will be placed at it's own optimized position with a bit of noise added
                    # # shape is should be (num_patches, batch_size, 2, 3)
                    # noisy_transformations = torch.stack([gen_noisy_transformations(len(batch), scale_factor, tx, ty) for scale_factor, tx, ty in zip(*position.transpose(1, 0))])
                    # # print(noisy_transformations.shape)
                    # #patch_batch = torch.cat([patch_t for _ in range(len(batch))])
                    
                    # patch_batches = torch.cat([x.repeat(len(batch), 1, 1, 1) for x in patches_t]) # get batch_sized batches of each patch in patches, size should be batch_size*num_patches
                    # batch_multi = batch.clone().repeat(len(patches_t), 1, 1, 1)
                    # transformations_multi = noisy_transformations.view(len(patches_t)*len(batch), 2, 3) # reshape transformation matrices
                    # #print(transformations_multi.shape)

                    # mod_img = place_patch(batch_multi, patch_batches, transformations_multi) 



                    mod_img *= 255. # convert input images back to range [0-255.]

                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patches.device)
                    # restrict patch+background to stay in range (0., 255.)
                    mod_img.clamp_(0., 255.)

                    # predict x, y, z, yaw
                    x, y, z, phi = model(mod_img)

                    # target_losses.append(torch.mean(all_l2))
                    # prepare shapes for MSE loss
                    pred = torch.stack([x, y, z])
                    pred = pred.squeeze(2).mT     # all batch_size*num_patches predictions in consecutive order

                    pred_v = pred.view(len(patches_t), len(batch), 3) # size : num_patches, batch_size, 3

                     # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                    mask = torch.isnan(target)
                    target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)

                    # old version
                    target_batch = (pred * mask) + target

                    target_losses.append(mse_loss(target_batch, pred))

                    # new version
                    # target_batch = (pred_v * mask) + target

                    #  # target_losses.append(mse_loss(target_batch, pred))

                    # #target_losses.append(torch.min(mse_loss(target_batch, pred_v)))
                    # target_loss = torch.stack([mse_loss(tar, pred) for tar, pred in zip(target_batch, pred_v)]) # calc mse for each of the predictions of each patch
                    # stats[:, k] += target_loss.detach().cpu().numpy()
                    # #print(target_loss)
                    # # # variant1
                    # # target_losses.append(torch.min(target_loss)) # keep only the minimum loss

                    # # variant2
                    # probabilities = torch.nn.functional.softmin(target_loss, dim=0)
                    # stats_p[:, k] += probabilities.detach().cpu().numpy()
                    # # expectation = probabilities.dot(target_loss)
                    # # target_losses.append(expectation)
                    # target_losses.append(torch.sum(target_loss))

                loss = torch.sum(torch.stack(target_losses))    # sum for all K targets
                # 7(loss)
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patches_t.data.clamp_(0., 1.)
            actual_loss /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            print("stats loss:", stats/len(dataset))
            print("stats probabilities:", stats_p/len(dataset))
            if actual_loss < best_loss:
                best_patch = patches_t.clone().detach()
                best_position = positions_t.clone().detach()
                best_loss = actual_loss
                best_stats = stats / len(dataset)
                best_stats_p = stats_p / len(dataset)
        
    except KeyboardInterrupt:
        print("Aborting optimization...")

    return best_patch, best_loss, best_position, best_stats, best_stats_p



if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', default='settings.yaml')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    # SETTINGS
    with open(args.file) as f:
        settings = yaml.load(f, Loader=yaml.FullLoader)

    # save settings directly to result folder for later use
    path = Path(settings['path'])
    os.makedirs(path, exist_ok = True)
    with open(path / 'settings.yaml', 'w') as f:
        yaml.dump(settings, f)

    lr_pos = settings['lr_pos']
    lr_patch = settings['lr_patch']
    num_hl_iter = settings['num_hl_iter']
    num_pos_restarts = settings['num_pos_restarts']
    num_pos_epochs = settings['num_pos_epochs']
    num_patch_epochs = settings['num_patch_epochs']
    batch_size = settings['batch_size']
    mode = settings['mode']
    quantized = settings['quantized']

    # get target values in correct shape and move tensor to device
    targets = [values for _, values in settings['targets'].items()]
    targets = np.array(targets, dtype=float).T
    targets = torch.from_numpy(targets).to(device).float()

    from util import load_dataset
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
    
    model.eval()

    train_set = load_dataset(path=dataset_path, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=0)
    # train_set.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # train_set.dataset.labels.to(device)
    
    test_set = load_dataset(path=dataset_path, batch_size=batch_size, shuffle=True, drop_last=False, train=False, num_workers=0)


    # # load the patch from misc folder
    # if settings['patch']['mode'] == 'face':
    #     patch_start = np.load(settings['patch']['path'])
    #     patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device) / 255.
    #     patch_start.clamp_(0., 1.)

    # # or start from a random patch
    # if settings['patch']['mode'] == 'random':
    #     patch_start = torch.rand(1, 1, 96, 160).to(device)

    # # or start from a white patch
    # if settings['patch']['mode'] == 'white':
    #     patch_start = torch.ones(1, 1, 300, 320).to(device)

    # multi patch, random only atm
    patches = torch.rand(1, 1, 96, 160).to(device)
    # patches[0,:,:,:] = torch.ones(1, 96, 160).to(device)

    optimization_pos_losses = []
    optimization_pos_vectors = []

    optimization_patches = []
    optimization_patch_losses = []

    train_losses = []
    test_losses = []

    # optimization_patches.append(patch_start)

    if mode == "split" or mode == "hybrid" or mode == "fixed":
        positions = torch.FloatTensor(len(targets), 3, 1).uniform_(-1., 1.).to(device)
    else:
        # start with placing the patch in the middle
        scale_factor, tx, ty = torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device)
        positions = []
        for target_idx in range(len(targets)):
            positions.append(torch.stack([scale_factor, tx, ty]))
        positions = torch.stack(positions)

    positions = positions.repeat(len(patches), 1, 1, 1)  # repeat initial positions for amount of patches

    # positions = torch.FloatTensor(len(patches), len(targets), 3, 1).uniform_(-1., 1.).to(device)


    optimization_pos_vectors.append(positions)

    # patch = patch_start.clone()

    # figures = []

    with PdfPages(Path(path) / 'result.pdf') as pdf:
        train_losses = []
        stats = []
        stats_p = []

        for train_iteration in trange(num_hl_iter):
            patches, loss_patch, positions, stat, stat_p = multi_joint(train_set, patches, model, optimization_pos_vectors[-1], targets=targets, lr=lr_patch, epochs=num_patch_epochs, path=path)
            train_losses.append(loss_patch.detach().cpu().numpy())
            stats.append(stat)
            stats_p.append(stat_p)
            #print(patches.shape, loss_patch.shape, positions.shape)
            print(positions)

            image, _ =  train_set.dataset.__getitem__(0)
            image = image.unsqueeze(0).to(device) / 255.

            figures_per_target = []
            fig, axs = plt.subplots(len(patches), len(targets), squeeze=False)
            for target_idx, (target, position) in enumerate(zip(targets, positions)):
                #print(position.shape)
            
                images_per_target = []
                for patch_idx, (patch, pos) in enumerate(zip(patches, position)):
                    #print(patch.shape, pos.shape)
                    sf_norm, tx_norm, ty_norm = norm_transformation(*pos)
                    print(sf_norm, tx_norm, ty_norm)

                    mod_img = place_patch(image, patch.unsqueeze(0), get_transformation(sf_norm, tx_norm, ty_norm), random_perspection=False)
                    axs[patch_idx, target_idx].imshow(mod_img.detach().cpu().numpy()[0][0], cmap='gray')
                    #images_per_target.append(mod_img.detach().cpu().numpy()[0][0])


            pdf.savefig(fig)
            plt.close(fig)        
                # fig, axs = plt.subplots(len()
                # for i, img in enumerate(images_per_target):
                #     axs[i].imshow(img, cmap='gray')

                # figures_per_target.append(fig)

        # plot individual losses over iterations
        stats = np.asarray(stats)
        fig, axs = plt.subplots(len(patches), len(targets), sharex=True, sharey=True, squeeze=False)
        fig.suptitle('Individual Training Losses')
        for target_idx in range(len(targets)):
            for patch_idx in range(len(patches)):
                axs[patch_idx, target_idx].plot(stats[:,patch_idx,target_idx])
        pdf.savefig(fig)
        plt.close(fig)

        # plot individual probs over iterations
        stats_p = np.asarray(stats_p)
        fig, axs = plt.subplots(len(patches), len(targets), sharex=True, sharey=True, squeeze=False)
        fig.suptitle('Individual Probabilities')
        for target_idx in range(len(targets)):
            for patch_idx in range(len(patches)):
                axs[patch_idx, target_idx].plot(stats_p[:,patch_idx,target_idx])
        pdf.savefig(fig)
        plt.close(fig)

        # plot individual expectation over iterations
        fig, axs = plt.subplots(len(patches), len(targets), sharex=True, sharey=True, squeeze=False)
        fig.suptitle('Individual Expectation')
        for target_idx in range(len(targets)):
            for patch_idx in range(len(patches)):
                axs[patch_idx, target_idx].plot(stats_p[:,patch_idx,target_idx] * stats[:,patch_idx,target_idx])
        pdf.savefig(fig)
        plt.close(fig)

        # Overall loss
        fig, ax = plt.subplots(1, 1)
        fig.suptitle('Training Loss')
        ax.plot(train_losses)
        pdf.savefig(fig)
        plt.close(fig)


        # for fig in figures:
            
            # fig.close()
        # print("2:", len(figures))
        # for epoch_figure in figures:
        #     # print("4:", len(epoch_figure))
        #     for target_figure in epoch_figure:
        #         pdf.savefig(target_figure)
