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

def gen_noisy_transformations(batch_size, sf, tx, ty):
    noisy_transformation_matrix = []
    for i in range(batch_size):
        sf_n = sf + np.random.normal(0.0, 0.1)
        tx_n = tx + np.random.normal(0.0, 0.1)
        ty_n = ty + np.random.normal(0.0, 0.1)

        scale_norm, tx_norm, ty_norm = norm_transformation(sf_n, tx_n, ty_n)
        single_matrix = get_transformation(scale_norm, tx_norm, ty_norm)
        noisy_transformation_matrix.append(single_matrix)
    
    return torch.cat(noisy_transformation_matrix)

def targeted_attack_joint(dataset, patch, model, target, lr=3e-2, epochs=10, path="eval/"):

    patch_t = patch.clone().requires_grad_(True)
    tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
    ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
    scaling_factor = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
    opt = torch.optim.Adam([patch_t, tx, ty, scaling_factor], lr=lr)

    #optimized_patches = []
    losses = []

    try:
        best_loss = np.inf

        for epoch in range(epochs):

            actual_loss = torch.tensor(0.).to(patch.device)
            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device)

                noisy_transformations = gen_noisy_transformations(len(batch), scaling_factor, tx, ty)
                patch_batch = torch.cat([patch_t for _ in range(len(batch))])

                mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations)

                # add noise to patch+background
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
                # restrict patch+background to stay in range (0., 255.)
                mod_img.clamp_(0., 255.)

                # predict x, y, z, yaw
                x, y, z, phi = model(mod_img)

                # calculate mean l2 losses (target, y) for all images in batch
                all_l2 = torch.sqrt(((y-target)**2)) 
                loss = torch.mean(all_l2)
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 255.)
            actual_loss /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            if actual_loss < best_loss:
                best_patch = patch_t.clone().detach()
                best_tx = tx.clone().detach()
                best_ty = ty.clone().detach()
                best_scaling = scaling_factor.clone().detach()
                best_loss = actual_loss
        
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    return best_patch, best_loss, best_scaling, best_tx, best_ty

def targeted_attack_patch(dataset, patch, model, scale_factor, tx, ty, target, lr=3e-2, epochs=10, path="eval/"):

    patch_t = patch.clone().requires_grad_(True)
    opt = torch.optim.Adam([patch_t], lr=lr)

    #optimized_patches = []
    losses = []

    try:
        best_loss = np.inf

        for epoch in range(epochs):

            actual_loss = torch.tensor(0.).to(patch.device)
            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device)

                #optimized_patches.append(patch_t.clone().detach())

                # scale_factor_n = scale_factor + np.random.normal(0.0, 0.1)
                # tx_n = tx + np.random.normal(0.0, 0.1)
                # ty_n = ty + np.random.normal(0.0, 0.1)
                #scale_norm, tx_norm, ty_norm = norm_transformation(scale_factor_n, tx_n, ty_n)

                #transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)
                noisy_transformations = gen_noisy_transformations(len(batch), scale_factor, tx, ty)
                patch_batch = torch.cat([patch_t for _ in range(len(batch))])

                mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations)

                # add noise to patch+background
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
                # restrict patch+background to stay in range (0., 255.)
                mod_img.clamp_(0., 255.)

                # predict x, y, z, yaw
                x, y, z, phi = model(mod_img)

                # calculate mean l2 losses (target, y) for all images in batch
                #all_l2 = torch.stack([torch.dist(target, i, p=2) for i in prediction[..., 1]])
                all_l2 = torch.sqrt(((y-target)**2)) 
                loss = torch.mean(all_l2)
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 255.)

                #optimized_patches.append(patch_t.clone().detach())
            actual_loss /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            if actual_loss < best_loss:
                best_patch = patch_t.clone().detach()
                best_loss = actual_loss
        
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    losses = torch.stack(losses)
    # optimized_patches = torch.stack(optimized_patches)

    # best_idx = torch.argmin(losses)
    # best_patch = optimized_patches[best_idx]

    # print(f"Found best patch at training step {best_idx}, loss: {losses[best_idx]}")

    # np.save(path+'best_patch.npy', best_patch)

    # np.save(path+'losses.npy', losses)

    return best_patch, best_loss#, losses

def targeted_attack_position(dataset, patch, model, target, lr=3e-2, include_start=False, tx_start=0., ty_start=0., sf_start=0.1, num_restarts=50, epochs=5, path="eval/targeted/"): 
    try: 
        best_loss = np.inf

        for restart in range(num_restarts):
            if include_start and restart == 0:
                # start with previously optimized values, fine-tuning
                tx = tx_start.clone().to(patch.device).requires_grad_(True)
                ty = ty_start.clone().to(patch.device).requires_grad_(True)
                scaling_factor = sf_start.clone().to(patch.device).requires_grad_(True)
            else:
                # start with random values 
                tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
                ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
                scaling_factor = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)

            # simple version with sampling, only
            # scale_norm, tx_norm, ty_norm = norm_transformation(scaling_factor, tx, ty)
            # transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

            # train_loss = calc_eval_loss(train_set, patch, transformation_matrix, model, target)
            # test_loss = calc_eval_loss(test_set, patch, transformation_matrix, model, target)

            # print("restart {} ({} {} {}) loss {} {}".format(restart, tx_norm.item(), ty_norm.item(), scale_norm.item(), train_loss, test_loss))
            # if train_loss < best_loss:
            #     best_loss = train_loss
            #     best_tx = tx.clone().detach()
            #     best_ty = ty.clone().detach()
            #     best_scaling = scaling_factor.clone().detach()
            
            opt = torch.optim.Adam([scaling_factor, tx, ty], lr=lr)
            
            for epoch in range(epochs):

                actual_loss = torch.tensor(0.).to(patch.device)
                for _, data in enumerate(dataset):
                    batch, _ = data
                    batch = batch.to(patch.device)
            
                    noisy_transformations = gen_noisy_transformations(len(batch), scaling_factor, tx, ty)
                    patch_batch = torch.cat([patch for _ in range(len(batch))])
                    mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations)
                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)    
                    mod_img.clamp_(0., 255.)

                    x, y, z, phi = model(mod_img)

                    # calculate mean l2 losses (target, y) for all images in batch
                    all_l2 = torch.sqrt(((y-target)**2))
                    loss = torch.mean(all_l2)
                    actual_loss += loss.clone().detach()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                actual_loss /= len(dataset)
                print("restart {} epoch {} loss {}".format(restart, epoch, actual_loss))
                if actual_loss < best_loss:
                    best_loss = actual_loss
                    best_tx = tx.clone().detach()
                    best_ty = ty.clone().detach()
                    best_scaling = scaling_factor.clone().detach()

    except KeyboardInterrupt:
        print("Aborting optimization...")

    return best_scaling, best_tx, best_ty, best_loss

def calc_eval_loss(dataset, patch, transformation_matrix, model, target):
    actual_loss = torch.tensor(0.).to(patch.device)

    for _, data in enumerate(dataset):
        batch, _ = data
        batch = batch.to(patch.device)

        mod_img = place_patch(batch, patch, transformation_matrix)
        mod_img.clamp_(0., 255.)

        x, y, z, phi = model(mod_img)
        all_l2 = torch.sqrt(((y-target)**2))
        loss = torch.mean(all_l2)
        actual_loss += loss.clone().detach()
    
    actual_loss /= len(dataset)

    return actual_loss



if __name__=="__main__":
    import os

    # TODO: multiple attacks per patch
    #  targets = [-2, 2]  # i.e., this patch should be possible to use to attack for y=-2 *and* y=2; we find *one* patch and *two* tx/ty/scaling results
    #  approach
    #  a) change targeted_attack_patch to get tx/ty/scaling per target; the loss is the sum of the individual losses
    #  b) call targeted_attack_position once for each target

    # SETTINGS
    path = 'eval/debug/settings0/'
    lr_pos = 1e-2
    lr_patch = 1e-1
    num_hl_iter = 5
    num_pos_restarts = 2
    num_pos_epochs = 1
    num_patch_epochs = 5
    mode = "regular" # regular or joint

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
    
    test_set = load_dataset(path=dataset_path, batch_size=20, shuffle=True, drop_last=False, train=False, num_workers=0)


    os.makedirs(path, exist_ok = True)

    # load the patch from misc folder
    patch_start = np.load("misc/custom_patch_resized.npy")
    patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device)

    # or start with a random patch
    # patch_start = torch.rand(1, 1, 200, 200).to(device) * 255.

    # or start from a white patch
    # patch_start = torch.ones(1, 1, 200, 200).to(device) * 255.



    # define target # TODO: currently only the y-value can be targeted
    target = torch.tensor(-2.0).to(device)

    optimization_pos_losses = []
    optimization_pos_vectors = []

    optimization_patches = []
    optimization_patch_losses = []

    train_losses = []
    test_losses = []

    optimization_patches.append(patch_start)

    # start with placing the patch in the middle
    scale_factor, tx, ty = torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device)

    patch = patch_start.clone()
    
    for train_iteration in trange(num_hl_iter):
        
        if mode != "joint":
            print("Optimizing patch...")
            patch, loss_patch = targeted_attack_patch(train_set, patch, model, scale_factor, tx, ty, target=target, lr=lr_patch, epochs=num_patch_epochs, path=path)
        else:
            patch, loss_patch, scale_factor, tx, ty = targeted_attack_joint(train_set, patch, model, target=target, lr=lr_patch, epochs=num_patch_epochs, path=path)
            loss_pos = loss_patch

        optimization_patches.append(patch.clone())
        optimization_patch_losses.append(loss_patch)

        if mode != "joint":

            print("Optimizing position...")
            scale_factor, tx, ty, loss_pos = targeted_attack_position(train_set, patch, model, target, include_start=True, tx_start=tx, ty_start=ty, sf_start=scale_factor, lr=lr_pos, num_restarts=num_pos_restarts, epochs=1, path=path)
        
        optimization_pos_vectors.append(torch.stack([scale_factor, tx, ty]))
        optimization_pos_losses.append(loss_pos)

        scale_norm, tx_norm, ty_norm = norm_transformation(scale_factor, tx, ty)
        transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

        train_loss = calc_eval_loss(train_set, patch, transformation_matrix, model, target)
        train_losses.append(train_loss)
        test_loss = calc_eval_loss(test_set, patch, transformation_matrix, model, target)
        test_losses.append(test_loss)

    #print(optimization_patch_losses)
    optimization_patches = torch.stack(optimization_patches)
    optimization_patch_losses = torch.stack(optimization_patch_losses)
    
    #print(optimization_pos_vectors)
    optimization_pos_vectors = torch.stack(optimization_pos_vectors)
    optimization_pos_losses = torch.stack(optimization_pos_losses)
    
    train_losses = torch.stack(train_losses)
    test_losses = torch.stack(test_losses)

    # print("patches shape: ", optimization_patches.shape)
    # print("patch losses shape: ", optimization_patch_losses.shape)

    # print("vectors shape: ", optimization_pos_vectors.shape)
    # print("pos losses shape: ", optimization_pos_losses.shape)

    print("Saving results...")
    # prepare data for plots
    #all_y = optimization_pos_vectors[..., 3]
    # normalize scale factor, tx and ty for plots
    norm_optimized_vecs = [norm_transformation(optimization_pos_vectors[i, 0], optimization_pos_vectors[i, 1], optimization_pos_vectors[i, 2]) for i in range(optimization_pos_vectors.shape[0])]
    
    all_sf = torch.stack([norm_optimized_vecs[i][0] for i in range(len(norm_optimized_vecs))])
    all_tx = torch.stack([norm_optimized_vecs[i][1] for i in range(len(norm_optimized_vecs))])
    all_ty = torch.stack([norm_optimized_vecs[i][2] for i in range(len(norm_optimized_vecs))])


    # save all results in numpy arrays for later use
    np.save(path+'patches.npy', optimization_patches.squeeze(1).squeeze(1).cpu().numpy())
    np.save(path+'patch_losses.npy', optimization_patch_losses.cpu().numpy())
    np.save(path+'positions.npy', np.array([all_sf.cpu().numpy(), all_tx.cpu().numpy(), all_ty.cpu().numpy()]))
    # np.save(path+'predictions_y.npy', all_y.cpu().numpy())
    np.save(path+'position_losses.npy', optimization_pos_losses.cpu().numpy())


    # evaluation on test set
    # TODO: needs to move inside loop...
    print("Evaluation...")
    scale_norm, tx_norm, ty_norm = norm_transformation(scale_factor, tx, ty)
    transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

    test_batch, test_gt = test_set.dataset[:]
    test_batch = test_batch.to(device)

    _, pred_base, _, _ = model(test_batch.float())

    mod_img = place_patch(test_batch, patch_start, transformation_matrix)
    mod_img.clamp_(0., 255.)
    _, pred_start_patch, _, _ = model(mod_img.float())

    mod_img = place_patch(test_batch, patch, transformation_matrix)
    mod_img.clamp_(0., 255.)

    _, pred_opt_patch, _, _ = model(mod_img.float())
    #pred_y = prediction[..., 1].detach().cpu().numpy()

    #rel_y = pred_y + target.cpu().numpy()

    boxplot_data = [pred_base.squeeze(1).detach().cpu().numpy(), pred_start_patch.squeeze(1).detach().cpu().numpy(), pred_opt_patch.squeeze(1).detach().cpu().numpy()]

    # vline_idx_patch = [i*len(loss_patch) for i in range(1, 10)]

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
        #ax.vlines(vline_idx_patch, 0, 1, transform=ax.get_xaxis_transform(), colors='r')
        ax.set_xlabel('iteration')
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
        ax.set_xlabel('iteration')
        ax.set_ylabel('mean l2 distance')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Losses after each patch & position optimization')
        ax.plot(train_losses.view(-1).cpu(), label='train set')
        ax.plot(test_losses.view(-1).cpu(), label='test set')
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

        fig, ax = plt.subplots(1, 1)
        ax.boxplot(boxplot_data, 1, 'D', labels=['base images', 'starting patch', 'optimized patch'])
        ax.set_title('boxplots for y, patches placed at optimal position')
        ax.set_ylabel('y')
        # axs[0].set_title('base images')
        # axs[1].boxplot(rel_y, 1, 'D')
        # axs[1].set_title('y - target y')
        pdf.savefig(fig)
        plt.close(fig)