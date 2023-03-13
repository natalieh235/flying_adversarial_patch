import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

from util import plot_saliency

def get_transformation(sf, tx, ty):
    translation_vector = torch.stack([tx, ty]).unsqueeze(0)

    eye = torch.eye(2, 2).unsqueeze(0).to(sf.device)
    rotation_matrix = eye * sf

    transformation_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)
    return transformation_matrix.float()

def norm_transformation(sf, tx, ty):
    tx_tanh = torch.tanh(tx)
    ty_tanh = torch.tanh(ty)
    scaling_norm = 0.1 * (torch.tanh(sf) + 1) + 0.3 # normalizes scaling factor to range [0.3, 0.5]

    return scaling_norm, tx_tanh, ty_tanh


def targeted_attack_patch(dataset, patch, model, target, transformation_matrix, lr=3e-2, path="eval/"):

    patch_t = patch.clone().requires_grad_(True)
    opt = torch.optim.Adam([patch_t], lr=lr)

    optimized_patches = []
    losses = []

    try: 

        for epoch in range(10):

            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device)

                optimized_patches.append(patch_t.clone().detach())

                mod_img = place_patch(batch.clone(), patch_t, transformation_matrix)
                # add noise to patch+background
                #mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
                # restrict patch+background to stay in range (0., 255.)
                mod_img.clamp_(0., 255.)

                # predict x, y, z, yaw
                prediction = torch.stack(model(mod_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

                # calculate mean l2 losses (target, y) for all images in batch
                #all_l2 = torch.stack([torch.dist(target, i, p=2) for i in prediction[..., 1]])
                all_l2 = torch.sqrt(((prediction[..., 1]-target)**2)) # got rid of slower list comprehension
                loss = torch.mean(all_l2)

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 255.)

                optimized_patches.append(patch_t.clone().detach())
        
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    losses = torch.stack(losses)
    optimized_patches = torch.stack(optimized_patches)

    best_idx = torch.argmin(losses)
    best_patch = optimized_patches[best_idx]

    print(f"Found best patch at training step {best_idx}, loss: {losses[best_idx]}")

    # np.save(path+'best_patch.npy', best_patch)

    # np.save(path+'losses.npy', losses)

    return best_patch, losses

def targeted_attack_position(dataset, patch, model, target, lr=3e-2, random=True, tx_start=0., ty_start=0., sf_start=0.1, num_restarts=50, path="eval/targeted/"): 
    # get a batch consisting of all images in the dataset
    batch, _ = dataset.dataset[:1000]
    batch = batch.to(patch.device)
    

    all_optimized = []
    all_losses = []

    # eye = torch.eye(2, 2).unsqueeze(0).to(patch.device)

    try: 
        for restart in trange(num_restarts):
            #if random:
                # start with random values 
            tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
            ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
            scaling_factor = torch.FloatTensor(1,).uniform_(0.3, 0.5).to(patch.device).requires_grad_(True)
            #else:
                # start with previously optimized values, fine-tuning
                # tx = tx_start.clone().to(patch.device).requires_grad_(True)
                # ty = ty_start.clone().to(patch.device).requires_grad_(True)
                # scaling_factor = sf_start.clone().to(patch.device).requires_grad_(True)
            
            opt = torch.optim.Adam([scaling_factor, tx, ty], lr=lr)

            optimized_vec = []
            losses = []
            
            # optimize for 200 training steps
            for i in range(200):
                scaling_norm, tx_tanh, ty_tanh = norm_transformation(scaling_factor, tx, ty)
                transformation_matrix = get_transformation(sf=scaling_norm, tx=tx_tanh, ty=ty_tanh)

                mod_img = place_patch(batch, patch, transformation_matrix)
                # add noise to patch+background
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)    
                mod_img.clamp_(0., 255.)

                prediction = torch.stack(model(mod_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

                # save sf, tx, ty and mean y values for later plots
                optimized_vec.append(torch.cat([scaling_factor.clone().detach(), tx.clone().detach(), ty.clone().detach(), torch.mean(prediction[..., 1]).clone().detach().unsqueeze(0)]))

                # calculate mean l2 losses (target, y) for all images in batch
                all_l2 = torch.sqrt(((prediction[..., 1]-target)**2)) # got rid of slower list comprehension
                loss = torch.mean(all_l2)
                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

            
            optimized_vec.append(torch.cat([scaling_factor.clone().detach(), tx.clone().detach(), ty.clone().detach(), torch.mean(prediction[..., 1]).clone().detach().unsqueeze(0)]))
            all_optimized.append(torch.stack(optimized_vec))
            all_losses.append(torch.stack(losses))

    except KeyboardInterrupt:
        print("Aborting optimization...")    

    #all_optimized = np.array(all_optimized)
    #all_losses= np.array(all_losses)
    # print(torch.tensor(all_optimized).shape)
    # print(torch.tensor(all_losses).shape)
    all_optimized = torch.stack(all_optimized)
    all_losses = torch.stack(all_losses)

    # find lowest y and get best run index
    best_run, lowest_idx = torch.argwhere(all_optimized[..., 3] == torch.min(all_optimized[..., 3]))[0]

    lowest_scale, lowest_tx, lowest_ty, _ = all_optimized[best_run, lowest_idx]
    #np.save(path+'best_transformation.npy', all_optimized[best_run, lowest_idx][:3])

    return lowest_scale.unsqueeze(0), lowest_tx.unsqueeze(0), lowest_ty.unsqueeze(0), all_losses[best_run], all_optimized[best_run]


if __name__=="__main__":
    import os

    from util import load_dataset, load_model
    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    train_set = load_dataset(path=dataset_path, batch_size=20, shuffle=True, drop_last=False, num_workers=0)
    # train_set.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # train_set.dataset.labels.to(device)

    path = 'eval/debug/pos/'
    os.makedirs(path, exist_ok = True)

    # load the patch from misc folder
    patch_start = np.load("misc/custom_patch_resized.npy")
    patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device)

    # or start with a random patch
    # patch_start = torch.rand(1, 1, 200, 200).to(device) * 255.

    # or start from a white patch
    # patch_start = torch.ones(1, 1, 200, 200).to(device) * 255.

    # set learning rate for the position and the optimization of the patch
    lr_pos = 3e-2
    lr_patch = 1e-1

    # define target # TODO: currently only the y-value can be targeted
    target = torch.tensor(-2.0).to(device)

    optimization_pos_losses = []
    optimization_pos_vectors = []

    optimization_patches = []
    optimization_patch_losses = []

    optimization_patches.append(patch_start)

    # calculate initial optimal patch position on 50 random restarts
    # TODO: parallelize restarts for multiple CPU cores
    print("Optimizing initial patch position...")
    scale_factor, tx, ty, loss_pos, optimized_vectors = targeted_attack_position(train_set, patch_start, model, target, lr=lr_pos, num_restarts=50, path=path)
    print(optimized_vectors.shape, loss_pos.shape)
    optimization_pos_vectors.append(optimized_vectors)
    optimization_pos_losses.append(loss_pos)

    scale_norm, tx_norm, ty_norm = norm_transformation(scale_factor, tx, ty)

    print(f"Optimized position: sf={scale_norm}, tx={tx_norm}, ty={ty_norm}")

    # calculate transformation matrix from single values
    transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

    patch = patch_start.clone()

    # decrease position learning rate for fine tuning
    #lr_pos = 1e-3
    for train_iteration in trange(10):
        
        print("Optimizing patch...")
        patch, loss_patch = targeted_attack_patch(train_set, patch, model, target=target, transformation_matrix=transformation_matrix, lr=lr_patch, path=path)

        optimization_patches.append(patch)
        optimization_patch_losses.append(loss_patch)

        # patch = patch
    
        print("Optimizing position...")
        scale_factor, tx, ty, loss_pos, optimized_vectors = targeted_attack_position(train_set, patch, model, target, tx_start=tx, ty_start=ty, sf_start=scale_factor, lr=lr_pos, num_restarts=10, path=path)
        optimization_pos_vectors.append(optimized_vectors)
        optimization_pos_losses.append(loss_pos)

        scale_norm, tx_norm, ty_norm = norm_transformation(scale_factor, tx, ty)

        transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

    optimization_patches = torch.stack(optimization_patches)
    optimization_patch_losses = torch.stack(optimization_patch_losses)

    optimization_pos_vectors = torch.stack(optimization_pos_vectors)
    optimization_pos_losses = torch.stack(optimization_pos_losses)

    # print("patches shape: ", optimization_patches.shape)
    # print("patch losses shape: ", optimization_patch_losses.shape)

    # print("vectors shape: ", optimization_pos_vectors.shape)
    # print("pos losses shape: ", optimization_pos_losses.shape)

    print("Saving results...")
    # prepare data for plots
    all_y = optimization_pos_vectors[..., 3]
    # normalize scale factor, tx and ty for plots
    norm_optimized_vecs = [norm_transformation(optimization_pos_vectors[i,..., 0], optimization_pos_vectors[i,..., 1], optimization_pos_vectors[i,..., 2]) for i in range(optimization_pos_vectors.shape[0])]
    
    all_sf = torch.stack([norm_optimized_vecs[i][0] for i in range(len(norm_optimized_vecs))])
    all_tx = torch.stack([norm_optimized_vecs[i][1] for i in range(len(norm_optimized_vecs))])
    all_ty = torch.stack([norm_optimized_vecs[i][2] for i in range(len(norm_optimized_vecs))])


    # save all results in numpy arrays for later use
    np.save(path+'patches.npy', optimization_patches.squeeze(1).squeeze(1).cpu().numpy())
    np.save(path+'patch_losses.npy', optimization_patch_losses.cpu().numpy())
    np.save(path+'positions.npy', np.array([all_sf.cpu().numpy(), all_tx.cpu().numpy(), all_ty.cpu().numpy()]))
    np.save(path+'predictions_y.npy', all_y.cpu().numpy())
    np.save(path+'position_losses.npy', optimization_pos_losses.cpu().numpy())


    # evaluation on test set
    print("Evaluation...")
    test_set = load_dataset(path=dataset_path, batch_size=403, shuffle=True, drop_last=False, train=False, num_workers=0)

    test_batch, test_gt = next(iter(test_set))
    test_batch = test_batch.to(device)

    pred_base = torch.stack(model(test_batch.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

    mod_img = place_patch(test_batch, patch_start, transformation_matrix)
    mod_img.clamp_(0., 255.)
    pred_start_patch = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

    mod_img = place_patch(test_batch, patch, transformation_matrix)
    mod_img.clamp_(0., 255.)

    pred_opt_patch = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
    #pred_y = prediction[..., 1].detach().cpu().numpy()

    #rel_y = pred_y + target.cpu().numpy()

    boxplot_data = [pred_base[..., 1].detach().cpu().numpy(), pred_start_patch[..., 1].detach().cpu().numpy(), pred_opt_patch[..., 1].detach().cpu().numpy()]

    vline_idx_patch = [i*len(loss_patch) for i in range(1, 10)]

    # create result pdf
    # get one image and ground-truth pose  
    base_img, ground_truth = train_set.dataset.__getitem__(0)
    base_img = base_img.unsqueeze(0).to(device)
    ground_truth = ground_truth.to(device)

    # get prediction for unaltered base image
    prediction = torch.stack(model(base_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

    # place the initial, unaltered patch in base image at the optimal position and get prediction
    mod_start = place_patch(base_img, patch_start, transformation_matrix)
    prediction_start = torch.stack(model(mod_start)).permute(1, 0, 2).squeeze(2).squeeze(0)

    # place the optimized patch in the image at the optimal position and get prediction
    mod_img = place_patch(base_img, patch, transformation_matrix)
    prediction_mod = torch.stack(model(mod_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

    # TODO: move to seperate function, maybe open file after each optimization and add figures
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pathlib import Path


    with PdfPages(Path(path) / 'result.pdf') as pdf:
        for idx, patch in enumerate(optimization_patches):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.set_title(f'Patch at training iteration {idx}')
            ax.imshow(patch.cpu()[0][0], cmap='gray')
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Loss patch optimization for all iterations, lr={lr_patch}')
        ax.plot(optimization_patch_losses.view(-1).cpu())
        ax.vlines(vline_idx_patch, 0, 1, transform=ax.get_xaxis_transform(), colors='r')
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean l2 distance')
        pdf.savefig(fig)
        plt.close(fig)

        # for idx in range(len(optimization_patch_losses)):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f'Loss patch optimization, iteration {idx}')
        #     ax.plot(optimization_patch_losses[idx].cpu())
        #     ax.set_xlabel('training steps')
        #     ax.set_ylabel('mean l2 distance')
        #     pdf.savefig(fig)
        #     plt.close(fig)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title(f'Predicted y for all iterations')
        # ax.plot(all_y.view(-1).cpu())
        # ax.set_xlabel('training steps')
        # ax.set_ylabel('y')
        # pdf.savefig(fig)
        # plt.close(fig)

        # for idx in range(len(all_y)):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f'Predicted y, iteration {idx}')
        #     ax.plot(all_y[idx].cpu())
        #     ax.set_xlabel('training steps')
        #     ax.set_ylabel('y')
        #     pdf.savefig(fig)
        #     plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Loss position optimization for all iterations, lr={lr_pos}')
        ax.plot(optimization_pos_losses.view(-1).cpu())
        ax.set_xlabel('training steps')
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
        ax.plot(all_sf.view(-1).cpu())
        ax.set_xlabel('training steps')
        ax.set_ylabel('scale factor')
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
        ax.plot(all_tx.view(-1).cpu())
        ax.set_xlabel('training steps')
        ax.set_ylabel('tx')
        pdf.savefig(fig)
        plt.close(fig)

        # for idx in range(all_tx.shape[0]):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f'tx, iteration {idx}')
        #     ax.plot(all_tx[idx].cpu())
        #     ax.set_xlabel('training steps')
        #     ax.set_ylabel('tx')
        #     pdf.savefig(fig)
        #     plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'ty for all iterations')
        ax.plot(all_ty.view(-1).cpu())
        ax.set_xlabel('training steps')
        ax.set_ylabel('ty')
        pdf.savefig(fig)
        plt.close(fig)

        # for idx in range(len(all_ty)):
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     ax.set_title(f'ty, iteration {idx}')
        #     ax.plot(all_ty[idx].cpu())
        #     ax.set_xlabel('training steps')
        #     ax.set_ylabel('ty')
        #     pdf.savefig(fig)
        #     plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Placed patch, after optimization, lr={lr_patch}')
        ax.imshow(mod_img[0][0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_saliency(base_img, ground_truth, model)
        fig.suptitle(f'y = {prediction[1].detach().cpu().item()}')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_saliency(mod_start, ground_truth, model)
        fig.suptitle(f'y = {prediction_start[1].detach().cpu().item()}')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_saliency(mod_img, ground_truth, model)
        fig.suptitle(f'y = {prediction_mod[1].detach().cpu().item()}')
        pdf.savefig(fig)
        plt.close(fig)

        fig, ax = plt.subplots(1, 1)
        ax.boxplot(boxplot_data, 1, 'D', labels=['base images', 'starting patch @ optimal position', 'optimized patch @ optimal position'])
        ax.set_title('boxplots for y')
        ax.set_ylabel('y')
        # axs[0].set_title('base images')
        # axs[1].boxplot(rel_y, 1, 'D')
        # axs[1].set_title('y - target y')
        pdf.savefig(fig)
        plt.close(fig)