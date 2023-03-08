import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

from util import plot_saliency

def gen_random_transformation(tx_min=-1, tx_max=1., ty_min=-1., ty_max=1., sf_min=0.1, sf_max=0.8):
    tx = torch.FloatTensor(1,).uniform_(tx_min, tx_max)
    ty = torch.FloatTensor(1,).uniform_(ty_min, ty_max)
    #sf = torch.FloatTensor(1,).uniform_(sf_min, sf_max)
    sf = torch.tensor(0.4)

    translation_vector = torch.stack([tx, ty]).unsqueeze(0)
    eye = torch.eye(2, 2).unsqueeze(0)
    rotation_matrix = eye * sf
    transformation_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)

    return transformation_matrix


def targeted_attack_patch(batch, patch, model, lr=3-2, path="eval/targeted/"):
    os.makedirs(path, exist_ok = True)
    
    patch_t = patch.clone().requires_grad_(True)
    opt = torch.optim.Adam([patch_t], lr=lr)

    target_low = torch.tensor(-2.0).to(patch.device)
    target_high = torch.tensor(2.0).to(patch.device)

    all_optimized_vec = []
    all_optimized_patches = []
    all_losses = []

    try: 
        transformation_low = gen_random_transformation(tx_max=0., ty_max=0.).to(patch.device)
        transformation_high = gen_random_transformation(tx_min=0., ty_min=0.).to(patch.device)

        #optimized_vec = []
        optimized_patches = []
        losses = []

        for i in trange(200):

            optimized_patches.append(patch_t.clone().detach()[0][0].cpu().numpy())

            mod_img_low = place_patch(batch.clone(), patch_t, transformation_low)
            # add noise to patch+background
            mod_img_low += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
            # restrict patch+background to stay in range (0., 255.)
            mod_img_low.clamp_(0., 255.)

            # predict x, y, z, yaw
            prediction_low = torch.stack(model(mod_img_low)).permute(1, 0, 2).squeeze(2).squeeze(0)

            #optimized_vec.append([scaling_sig.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction_mod[..., 1]).clone().detach().cpu().item()])

            # calculate mean l2 losses (target, y) for all images in batch
            all_l2 = torch.stack([torch.dist(target_low, i, p=2) for i in prediction_low[..., 1]])
            loss_low = torch.mean(all_l2)

            mod_img_high = place_patch(batch.clone(), patch_t, transformation_high)
            # add noise to patch+background
            mod_img_high += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
            # restrict patch+background to stay in range (0., 255.)
            mod_img_high.clamp_(0., 255.)

            # predict x, y, z, yaw
            prediction_high = torch.stack(model(mod_img_high)).permute(1, 0, 2).squeeze(2).squeeze(0)

            #optimized_vec.append([scaling_sig.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction_mod[..., 1]).clone().detach().cpu().item()])

            # calculate mean l2 losses (target, y) for all images in batch
            all_l2 = torch.stack([torch.dist(target_high, i, p=2) for i in prediction_high[..., 1]])
            loss_high = torch.mean(all_l2)

            loss = (0.5 * loss_low) + (0.5 * loss_high)


            losses.append(loss.clone().detach().cpu().item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            patch_t.data.clamp_(0., 255.)

        
        #optimized_vec.append([scaling_sig.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction_mod[..., 1]).clone().detach().cpu().item()])
        optimized_patches.append(patch_t.clone().detach()[0][0].cpu().numpy())

        losses = np.array(losses)
        optimized_patches = np.array(optimized_patches)
        #all_optimized_vec.append(np.array(optimized_vec))
        # all_optimized_patches.append(np.array(optimized_patches))
        # all_losses.append(np.array(losses))

        # all_optimized_a = np.array(all_optimized_vec)
        # all_optimized_patches_a = np.array(all_optimized_patches)
        # all_losses_a = np.array(all_losses)

    except KeyboardInterrupt:
        print("Aborting optimization...")    

    #best_round, lowest_idx = np.argwhere(all_optimized_a[..., 3] == np.min(all_optimized_a[..., 3]))[0]

    #best_patch = all_optimized_patches_a[best_round, lowest_idx]
    #lowest_scale, lowest_tx, lowest_ty, lowest_y = all_optimized_a[best_round, lowest_idx]

    #print("Best random iteration: ", best_round, lowest_idx)
    #print("Best scale factor & translation vector: ", lowest_scale, lowest_tx, lowest_ty)
    #print("Best prediciton: y = ", lowest_y)

    # print("--Shape sanity checks--")
    # print("patch shape: ", best_patch.shape)
    # print("all patches shape: ", all_optimized_patches_a.shape)
    # print()
    # print("best round optimized vec shape: ", all_optimized_a[best_round].shape)
    # print("all optimized vec shape: ", all_optimized_a.shape)

    best_idx = np.argmin(losses)
    best_patch = optimized_patches[best_idx]

    print(f"Found best patch at training step {best_idx}, loss: {losses[best_idx]}")

    np.save(path+'all_patches.npy', optimized_patches)
    np.save(path+'best_patch.npy', best_patch)


    # np.save(path+'all_optimized_vec.npy', all_optimized_a)
    # np.save(path+'best_optimized_vec.npy', all_optimized_a[best_round])

    np.save(path+'losses.npy', losses)

    return best_patch, losses

def targeted_attack_position(batch, patch, model, lr=3e-2, path="eval/targeted/"):
    target = torch.tensor(-2.0).to(patch.device)

    all_optimized = []
    all_losses = []

    eye = torch.eye(2, 2).unsqueeze(0).to(patch.device)

    try: 
        for restart in trange(50):
            tx = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
            ty = torch.FloatTensor(1,).uniform_(-1., 1.).to(patch.device).requires_grad_(True)
            scaling_factor = torch.FloatTensor(1,).uniform_(0.1, 0.8).to(patch.device).requires_grad_(True)

            opt = torch.optim.Adam([scaling_factor, tx, ty], lr=3e-2)

            optimized_vec = []
            losses = []

            for i in range(200):
                tx_tanh = torch.tanh(tx)
                ty_tanh = torch.tanh(ty)
                scaling_sig = torch.sigmoid(scaling_factor)

                translation_vector = torch.stack([tx_tanh, ty_tanh]).unsqueeze(0)
                rotation_matrix = eye * scaling_sig
                transformation_matrix = torch.cat((rotation_matrix, translation_vector), dim=2)

                mod_img = place_patch(batch, patch, transformation_matrix)
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape).to(patch.device)
                mod_img.clamp_(0., 255.)

                prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

                optimized_vec.append([scaling_sig.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction_mod[..., 1]).clone().detach().cpu().item()])

                all_l2 = torch.stack([torch.dist(target, i, p=2) for i in prediction_mod[..., 1]])
                loss = torch.mean(all_l2)
                losses.append(loss.clone().detach().cpu().item())

                #loss = torch.dist(target, prediction_mod[..., 1], p=2)

                opt.zero_grad()
                loss.backward()
                opt.step()

            
            optimized_vec.append([scaling_sig.clone().detach().cpu().item(), tx_tanh.clone().detach().cpu().item(), ty_tanh.clone().detach().cpu().item(), torch.mean(prediction_mod[..., 1]).clone().detach().cpu().item()])

            all_optimized.append(np.array(optimized_vec))
            all_losses.append(np.array(losses))

        all_optimized_a = np.array(all_optimized)
        all_losses_a = np.array(all_losses)

    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    lowest_plot, lowest_idx = np.argwhere(all_optimized_a[..., 3] == np.min(all_optimized_a[..., 3]))[0]

    lowest_scale, lowest_tx, lowest_ty, lowest_y = all_optimized_a[lowest_plot, lowest_idx]
    # print("Best loss: ", np.min(losses), np.argmin(losses))
    print("Best scale factor & translation vector: ", lowest_scale, lowest_tx, lowest_ty)
    print("Best prediciton: y = ", lowest_y)

    np.save(path+'best_optimized.npy', all_optimized_a[lowest_plot])
    np.save(path+'best_losses.npy', all_losses_a[lowest_plot])

    # np.save(path+'losses', losses)
    # np.save(path+'vectors', vectors)
    # np.save(path+'predicitons', predictions)

    return lowest_scale, lowest_tx, lowest_ty

if __name__=="__main__":
    # import matplotlib.pyplot as plt
    import os

    from util import load_dataset, load_model
    model_path = '/home/hanfeld/adversarial_frontnet/pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '/home/hanfeld/adversarial_frontnet/pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=200, shuffle=True, drop_last=True, num_workers=0)
    # dataset.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # dataset.dataset.labels.to(device)

    path = 'eval/debug/patch_bs_200/'

    patch_start = np.load("/home/hanfeld/adversarial_frontnet/misc/custom_patch_resized.npy")
    patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device)
    
    # patch_start = torch.FloatTensor(1, 1, 200, 200).uniform_(0., 255.).to(device)

    batch, gt = next(iter(dataset))
    batch = batch.to(device)

    patch_1e1, loss_1e1 = targeted_attack_patch(batch, patch_start, model, lr=1e-1, path=path+'1e-1/')
    #patch_5e2, loss_5e2 = targeted_attack_patch(batch, patch_start, model, lr=5e-2, path=path+'5e-2/')
    #patch_3e2, loss_3e2 = targeted_attack_patch(batch, patch_start, model, lr=3e-2, path=path+'3e-2/')
    #patch_1e2, loss_1e2 = targeted_attack_patch(batch, patch_start, model, lr=1e-2, path=path+'1e-2/')

    rand_transformation = gen_random_transformation(tx_min=0.1, tx_max=0.7, sf_min=0.3, sf_max=0.5)
    base_img = batch[0].unsqueeze(0)
    prediction = torch.stack(model(base_img)).permute(1, 0, 2).squeeze(2).squeeze(0)
    print(prediction)
    
    mod_img = place_patch(base_img, torch.tensor(patch_1e1).unsqueeze(0).unsqueeze(0).to(device), rand_transformation.to(device))
    prediction_mod = torch.stack(model(mod_img)).permute(1, 0, 2).squeeze(2).squeeze(0)
    print(prediction_mod)

    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pathlib import Path


    with PdfPages(Path(path) / 'result.pdf') as pdf:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Custom patch 1, base')
        ax.imshow(patch_start.detach().cpu().numpy()[0][0], cmap='gray')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Custom patch 1, after optimization, lr=1e-1')
        ax.imshow(patch_1e1, cmap='gray')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Loss for custom patch 1, lr=1e-1')
        ax.plot(loss_1e1)
        ax.set_xlabel('training steps')
        ax.set_ylabel('mean l2 distance')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_title('Placed patch, after optimization, lr=1e-1')
        ax.imshow(mod_img[0][0].detach().cpu().numpy(), cmap='gray')
        plt.axis('off')
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_saliency(base_img[0], gt[0].to(device), model) 
        pdf.savefig(fig)
        plt.close(fig)

        fig = plot_saliency(mod_img[0], gt[0].to(device), model) 
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
