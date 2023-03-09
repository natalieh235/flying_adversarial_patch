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


def targeted_attack_patch(dataset, patch, model, target, transformation_matrix, lr=3-2, path="eval/"):

    patch_t = patch.clone().requires_grad_(True)
    opt = torch.optim.Adam([patch_t], lr=lr)

    optimized_patches = []
    losses = []

    try: 

        for epoch in trange(10):

            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device)

                optimized_patches.append(patch_t.clone().detach()[0][0].cpu().numpy())

                mod_img = place_patch(batch.clone(), patch_t, transformation_matrix)
                # add noise to patch+background
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
                # restrict patch+background to stay in range (0., 255.)
                mod_img.clamp_(0., 255.)

                # predict x, y, z, yaw
                prediction = torch.stack(model(mod_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

                # calculate mean l2 losses (target, y) for all images in batch
                all_l2 = torch.stack([torch.dist(target, i, p=2) for i in prediction[..., 1]])
                loss = torch.mean(all_l2)

                losses.append(loss.clone().detach().cpu().item())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 255.)

                optimized_patches.append(patch_t.clone().detach()[0][0].cpu().numpy())
        

        losses = np.array(losses)
        optimized_patches = np.array(optimized_patches)

    except KeyboardInterrupt:
        print("Aborting optimization...")    

    best_idx = np.argmin(losses)
    best_patch = optimized_patches[best_idx]

    print(f"Found best patch at training step {best_idx}, loss: {losses[best_idx]}")

    np.save(path+'best_patch.npy', best_patch)

    np.save(path+'losses.npy', losses)

    return best_patch, losses

def targeted_attack_position(dataset, patch, model, target, lr=3e-2, random=True, tx_start=0., ty_start=0., sf_start=0.1, num_restarts=50, path="eval/targeted/"): 
    # get a random batch from the dataset
    batch, _ = next(iter(dataset))
    batch = batch.to(patch.device)
    # TODO: Not sure if optimizing on a single batch is ok in this case!

    all_optimized = []
    all_losses = []

    # eye = torch.eye(2, 2).unsqueeze(0).to(patch.device)

    try: 
        for restart in trange(num_restarts):
            if random:
                # start with random values 
                tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
                ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
                scaling_factor = torch.FloatTensor(1,).uniform_(0.3, 0.5).to(patch.device).requires_grad_(True)
            else:
                # start with previously optimized values, fine-tuning
                tx = tx_start.clone().to(patch.device).requires_grad_(True)
                ty = ty_start.clone().to(patch.device).requires_grad_(True)
                scaling_factor = sf_start.clone().to(patch.device).requires_grad_(True)
            
            opt = torch.optim.Adam([scaling_factor, tx, ty], lr=lr)

            optimized_vec = []
            losses = []
            
            # optimize for 200 training steps
            for i in range(200):
                tx_tanh = torch.tanh(tx)
                ty_tanh = torch.tanh(ty)
                scaling_norm = 0.1 * (torch.tanh(scaling_factor) + 1) + 0.3 # normalizes scaling factor to range [0.3, 0.5]

                transformation_matrix = get_transformation(sf=scaling_norm, tx=tx_tanh, ty=ty_tanh)

                mod_img = place_patch(batch, patch, transformation_matrix)
                # add noise to patch+background
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)    
                mod_img.clamp_(0., 255.)

                prediction = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

                # save sf, tx, ty and mean y values for later plots
                optimized_vec.append([scaling_norm.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction[..., 1]).clone().detach().cpu().item()])

                # calculate mean l2 losses (target, y) for all images in batch
                all_l2 = torch.stack([torch.dist(target, i, p=2) for i in prediction[..., 1]])
                loss = torch.mean(all_l2)
                losses.append(loss.clone().detach().cpu().item())

                opt.zero_grad()
                loss.backward()
                opt.step()

            
            optimized_vec.append([scaling_norm.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction[..., 1]).clone().detach().cpu().item()])
            all_optimized.append(optimized_vec)
            all_losses.append(losses)

    except KeyboardInterrupt:
        print("Aborting optimization...")    

    all_optimized = np.array(all_optimized)
    all_losses= np.array(all_losses)

    # find lowest y and get best run index
    best_run, lowest_idx = np.argwhere(all_optimized[..., 3] == np.min(all_optimized[..., 3]))[0]

    lowest_scale, lowest_tx, lowest_ty, _ = all_optimized[best_run, lowest_idx]
    np.save(path+'best_transformation.npy', all_optimized[best_run, lowest_idx][:3])

    return lowest_scale, lowest_tx, lowest_ty, all_losses[best_run], all_optimized[best_run]


if __name__=="__main__":
    import os

    from util import load_dataset, load_model
    model_path = 'pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = 'pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=20, shuffle=True, drop_last=True, num_workers=0)
    # dataset.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # dataset.dataset.labels.to(device)

    path = 'eval/debug/position&patch/'
    os.makedirs(path, exist_ok = True)

    # load the patch from misc folder
    patch_start = np.load("misc/custom_patch_resized.npy")
    patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device)

    # or start with a random patch
    # patch_start = torch.rand(1, 1, 200, 200).to(device)

    # or start from a white patch
    # patch_start = torch.ones(1, 1, 200, 200).to(device) * 255.

    # set learning rate for the position and the optimization of the patch
    lr_pos = 3e-2
    lr_patch = 1e-1

    # define target # TODO: currently only the y-value can be targeted
    target = torch.tensor(-2.0).to(device)

    # calculate initial optimal patch position on 50 random restarts
    # TODO: optimization is super slow, could be due to:
    # 1) moving values to cpu too often, 
    # 2) the if random at the beginning of each restart?
    print("Optimizing initial patch position...")
    scale_factor, tx, ty, loss_pos, optimized_vectors = targeted_attack_position(dataset, patch_start, model, target, lr=lr_patch, num_restarts=50, path=path)

    print(f"Optimized position: sf={scale_factor}, tx={tx}, ty={ty}")
    scale_factor = torch.tensor([scale_factor])
    tx = torch.tensor([tx])
    ty = torch.tensor([ty])

    # calculate transformation matrix from single values
    transformation_matrix = get_transformation(scale_factor, tx, ty).to(device)

    print("Optimizing patch on whole dataset for 10 epochs...")
    patch, loss_patch = targeted_attack_patch(dataset, patch_start, model, target=target, transformation_matrix=transformation_matrix, lr=lr_patch, path=path)

    #2nd iteration position
    print("Fine-tune position again on 10 restarts...")
    scale_factor_2, tx_2, ty_2, loss_pos_2, optimized_vectors_2 = targeted_attack_position(dataset, patch_start, model, target, random=False, tx_start=tx, ty_start=ty, sf_start=scale_factor, lr=lr_patch, num_restarts=10, path=path)
    print(f"Optimized position: sf={scale_factor}, tx={tx}, ty={ty}")
    scale_factor_2 = torch.tensor([scale_factor_2])
    tx_2 = torch.tensor([tx_2])
    ty_2 = torch.tensor([ty_2])
    transformation_matrix = get_transformation(scale_factor_2, tx_2, ty_2).to(device)

    # create result pdf
    # get one image and ground-truth pose  
    base_img, ground_truth = dataset.dataset.__getitem__(0)
    base_img = base_img.unsqueeze(0).to(device)
    ground_truth = ground_truth.to(device)

    # get prediction for unaltered base image
    prediction = torch.stack(model(base_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

    # place the initial, unaltered patch in base image at the optimal position and get prediction
    mod_start = place_patch(base_img, patch_start, transformation_matrix.to(device))
    prediction_start = torch.stack(model(mod_start)).permute(1, 0, 2).squeeze(2).squeeze(0)

    # place the optimized patch in the image at the optimal position and get prediction
    mod_img = place_patch(base_img, torch.tensor(patch).unsqueeze(0).unsqueeze(0).to(device), transformation_matrix.to(device))
    prediction_mod = torch.stack(model(mod_img)).permute(1, 0, 2).squeeze(2).squeeze(0)

    # TODO: move to seperate function, maybe open file after each optimization and add figures
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pathlib import Path


    with PdfPages(Path(path) / 'result.pdf') as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Custom patch, trainingstep 0')
        ax.imshow(patch_start.detach().cpu().numpy()[0][0], cmap='gray')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Loss for best position, lr={lr_pos}')
        ax.plot(loss_pos)
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean l2 distance')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('predicted mean y during training')
        ax.plot(optimized_vectors[..., 3])
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean y')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('optimization scale factor')
        ax.plot(optimized_vectors[..., 0])
        ax.set_xlabel('training steps')
        ax.set_ylabel('scale factor')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('optimization tx')
        ax.plot(optimized_vectors[..., 1])
        ax.set_xlabel('training steps')
        ax.set_ylabel('tx')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('optimization ty')
        ax.plot(optimized_vectors[..., 2])
        ax.set_xlabel('training steps')
        ax.set_ylabel('ty')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Custom patch, after optimization, lr={lr_patch}')
        ax.imshow(patch, cmap='gray')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Loss for custom patch 1, lr={lr_patch}')
        ax.plot(loss_patch)
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean l2 distance')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'Loss for best position, 2nd iteration, lr={lr_pos}')
        ax.plot(loss_pos_2)
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean l2 distance')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('predicted mean y during training')
        ax.plot(optimized_vectors_2[..., 3])
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean y')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('optimization scale factor')
        ax.plot(optimized_vectors_2[..., 0])
        ax.set_xlabel('training steps')
        ax.set_ylabel('scale factor')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('optimization tx')
        ax.plot(optimized_vectors_2[..., 1])
        ax.set_xlabel('training steps')
        ax.set_ylabel('tx')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('optimization ty')
        ax.plot(optimized_vectors_2[..., 2])
        ax.set_xlabel('training steps')
        ax.set_ylabel('ty')
        pdf.savefig(fig)
        plt.close(fig)

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



        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Custom patch 1, after optimization, lr=5e-2')
        # ax.imshow(patch_5e2, cmap='gray')
        # plt.axis('off')
        # pdf.savefig(fig)
        # plt.close(fig)


        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Loss for custom patch 1, lr=5e-2')
        # ax.plot(loss_5e2)
        # ax.set_xlabel('training steps')
        # ax.set_ylabel('mean l2 distance')
        # pdf.savefig(fig)
        # plt.close(fig)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Custom patch 1, after optimization, lr=3e-2')
        # ax.imshow(patch_3e2, cmap='gray')
        # plt.axis('off')
        # pdf.savefig(fig)
        # plt.close(fig)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Loss for custom patch 1, lr=3e-2')
        # ax.plot(loss_3e2)
        # ax.set_xlabel('training steps')
        # ax.set_ylabel('mean l2 distance')
        # pdf.savefig(fig)
        # plt.close(fig)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Custom patch 1, after optimization, lr=1e-2')
        # ax.imshow(patch_1e2, cmap='gray')
        # plt.axis('off')
        # pdf.savefig(fig)
        # plt.close(fig)

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_title('Loss for custom patch 1, lr=1e-2')
        # ax.plot(loss_1e2)
        # ax.set_xlabel('training steps')
        # ax.set_ylabel('mean l2 distance')
        # pdf.savefig(fig)
        # plt.close(fig)






### depracated
# def untargeted_attack(image, patch, model, transformation_matrix, path='eval/untargeted/'): # angle, scale, tx, ty,
#     # initialize optimizer
#     opt = torch.optim.Adam([transformation_matrix], lr=1e-1)
#     prediction = torch.concat(model(image)).squeeze(1)
    
    
#     new_image = place_patch(image, patch, transformation_matrix)
#     # pred_attack = torch.concat(model(new_image)).squeeze(1)
#     # loss = -torch.dist(prediction, pred_attack, p=2)

#     losses = []
#     # losses.append(loss.detach().numpy())

#     i = 0.
#     try:
#         while True:
#             i += 1
#             pred_attack= torch.concat(model(new_image)).squeeze(1)
#             loss = -torch.dist(prediction, pred_attack, p=2)
#             losses.append(loss.detach().numpy())
#             opt.zero_grad()
#             loss.backward(retain_graph=True)
#             opt.step()

#             # patch.data.clamp_(0., 255.)
#             new_image = place_patch(image, patch, transformation_matrix)
#             if i % 100 == 0:
#                 print("step %d, loss %.6f" % (i, loss))
#                 print("transformation matrix: ", transformation_matrix.detach().numpy())
#                 #print("step %d, loss %.6f, angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (i, loss, np.degrees(angle.detach().numpy()), scale.detach().numpy(), tx.detach().numpy(), ty.detach().numpy()))
#     except KeyboardInterrupt:
#         print("Aborting optimization...")    

#     print("Bing!")
#     print("Last loss: ", loss.detach().cpu().numpy())
#     print("Last prediciton: ", pred_attack)

#     np.save(path+'losses_test', losses)

#     return patch, transformation_matrix #[angle, scale, tx, ty]
