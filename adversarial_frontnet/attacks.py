import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

from util import plot_patch

def targeted_attack(batch, patch, model, path="eval/targeted/"):
    target = torch.tensor(-2.0)

    all_optimized = []
    all_losses = []

    eye = torch.eye(2, 2).unsqueeze(0)

    try: 
        for restart in trange(50):
            tx = torch.FloatTensor(1,).uniform_(-1., 1.).requires_grad_(True)
            ty = torch.FloatTensor(1,).uniform_(-1., 1.).requires_grad_(True)
            scaling_factor = torch.FloatTensor(1,).uniform_(0.1, 0.8).requires_grad_(True)

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

                mod_img = place_patch(batch, patch.unsqueeze(0).unsqueeze(0), transformation_matrix)
                mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch.shape)
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
        all_losses_a = np.array(all_losses_a)

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

    return patch, [lowest_scale, lowest_tx, lowest_ty]

def untargeted_attack(image, patch, model, transformation_matrix, path='eval/untargeted/'): # angle, scale, tx, ty,
    # initialize optimizer
    opt = torch.optim.Adam([transformation_matrix], lr=1e-1)
    prediction = torch.concat(model(image)).squeeze(1)
    
    
    new_image = place_patch(image, patch, transformation_matrix)
    # pred_attack = torch.concat(model(new_image)).squeeze(1)
    # loss = -torch.dist(prediction, pred_attack, p=2)

    losses = []
    # losses.append(loss.detach().numpy())

    i = 0.
    try:
        while True:
            i += 1
            pred_attack= torch.concat(model(new_image)).squeeze(1)
            loss = -torch.dist(prediction, pred_attack, p=2)
            losses.append(loss.detach().numpy())
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            # patch.data.clamp_(0., 255.)
            new_image = place_patch(image, patch, transformation_matrix)
            if i % 100 == 0:
                print("step %d, loss %.6f" % (i, loss))
                print("transformation matrix: ", transformation_matrix.detach().numpy())
                #print("step %d, loss %.6f, angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (i, loss, np.degrees(angle.detach().numpy()), scale.detach().numpy(), tx.detach().numpy(), ty.detach().numpy()))
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().cpu().numpy())
    print("Last prediciton: ", pred_attack)

    np.save(path+'losses_test', losses)

    return patch, transformation_matrix #[angle, scale, tx, ty]

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
    dataset = load_dataset(path=dataset_path, batch_size=20, shuffle=True, drop_last=True, num_workers=0)
    # dataset.dataset.data.to(device)   # TODO: __getitem__ and next(iter(.)) are still yielding data on cpu!
    # dataset.dataset.labels.to(device)

    path = 'eval/debugging_scale_trans/'
    os.makedirs(path, exist_ok = True)

    patch = np.load("/home/hanfeld/adversarial_frontnet/misc/custom_patch.npy")
    patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
    
    batch, _ = next(iter(dataset))
    batch = batch.to(device)

    patch, optimized_vecs = targeted_attack(batch, patch, model, path='eval/new/')