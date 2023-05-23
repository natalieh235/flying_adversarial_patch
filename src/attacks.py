import numpy as np
import os
import yaml
import argparse
import torch
from torch.nn.functional import mse_loss
from tqdm import trange

from patch_placement import place_patch

from pathlib import Path

def get_transformation(sf, tx, ty):
    translation_vector = torch.stack([tx, ty]).unsqueeze(0) #torch.zeros([1], device=tx.device)]).unsqueeze(0)

    eye = torch.eye(2, 2).unsqueeze(0).to(sf.device)
    scale = eye * sf

    # print(scale.shape, translation_vector.shape)

    transformation_matrix = torch.cat([scale, translation_vector], dim=2)
    return transformation_matrix.float()

def norm_transformation(sf, tx, ty):
    tx_tanh = torch.tanh(tx)
    ty_tanh = torch.tanh(ty)
    scaling_norm = 0.1 * (torch.tanh(sf) + 1) + 0.3 # normalizes scaling factor to range [0.3, 0.5]

    return scaling_norm, tx_tanh, ty_tanh

# def get_rotation(yaw, pitch, roll):
#     rotation_yaw = np.array([[np.cos(yaw), -np.sin(yaw), 0.0, 0.0],
#                          [np.sin(yaw), np.cos(yaw), 0.0, 0.0],
#                          [0.0, 0.0, 1.0, 0.0],
#                          [0.0, 0.0, 0.0, 1.0]])
    
#     rotation_pitch = np.array([[np.cos(pitch), 0.0, np.sin(pitch), 0.0],
#                            [0.0, 1.0, 0.0, 0.0],
#                            [-np.sin(pitch), 0.0, np.cos(pitch), 0.0],
#                            [0.0, 0.0, 0.0, 1.0]])

#     rotation_roll = np.array([[1.0, 0.0, 0.0, 0.0],
#                           [0.0, np.cos(roll), -np.sin(roll), 0.0],
#                           [0.0, np.sin(roll), np.cos(roll), 0.0],
#                           [0.0, 0.0, 0.0, 1.0]])     

#     rotation_matrix = rotation_yaw @ rotation_pitch @ rotation_roll
#     return rotation_matrix

def gen_noisy_transformations(batch_size, sf, tx, ty):
    noisy_transformation_matrix = []
    for i in range(batch_size):
        sf_n = sf + np.random.normal(0.0, 0.1)
        tx_n = tx + np.random.normal(0.0, 0.1)
        ty_n = ty + np.random.normal(0.0, 0.1)

        scale_norm, tx_norm, ty_norm = norm_transformation(sf_n, tx_n, ty_n)
        matrix = get_transformation(scale_norm, tx_norm, ty_norm)

        # random_yaw = np.deg2rad(np.random.normal(-10, 10))
        # random_pitch = np.deg2rad(np.random.normal(-5, 5))
        # random_roll = np.deg2rad(np.random.normal(-5, 5))
        #noisy_rotation = torch.tensor(get_rotation(random_yaw, random_pitch, random_roll)).float().to(matrix.device)

        #matrix[..., :3, :3] = noisy_rotation @ matrix[..., :3, :3]

        noisy_transformation_matrix.append(matrix)
    
    return torch.cat(noisy_transformation_matrix)

def targeted_attack_joint(dataset, patch, model, positions, targets, lr=3e-2, epochs=10, path="eval/"):

    patch_t = patch.clone().requires_grad_(True)
    positions_t = positions.clone().requires_grad_(True)

    opt = torch.optim.Adam([patch_t, positions_t], lr=lr)

    losses = []

    try:
        best_loss = np.inf

        for epoch in range(epochs):

            actual_loss = torch.tensor(0.).to(patch.device)
            for _, data in enumerate(dataset):
                batch, _ = data
                batch = batch.to(patch.device) / 255. # limit images to range [0-1]

                target_losses = []
                for position, target in zip(positions_t, targets):
                    scale_factor, tx, ty = position
                    noisy_transformations = gen_noisy_transformations(len(batch), scale_factor, tx, ty)
                    patch_batch = torch.cat([patch_t for _ in range(len(batch))])

                    mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations) 
                    mod_img *= 255. # convert input images back to range [0-255.]

                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patch.device)
                    # restrict patch+background to stay in range (0., 255.)
                    mod_img.clamp_(0., 255.)

                    # predict x, y, z, yaw
                    x, y, z, phi = model(mod_img)

                    # target_losses.append(torch.mean(all_l2))
                    # prepare shapes for MSE loss
                    # TODO: improve readbility!
                    pred = torch.stack([x, y, z])
                    pred = pred.squeeze(2).mT

                     # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                    mask = torch.isnan(target)
                    target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)
                    target_batch = (pred * mask) + target

                    target_losses.append(mse_loss(target_batch, pred))

                loss = torch.sum(torch.stack(target_losses))
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 1.)
            actual_loss /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            if actual_loss < best_loss:
                best_patch = patch_t.clone().detach()
                best_position = positions_t.clone().detach()
                best_loss = actual_loss
        
    except KeyboardInterrupt:
        print("Aborting optimization...")

    return best_patch, best_loss, best_position

def targeted_attack_patch(dataset, patch, model, positions, targets, lr=3e-2, epochs=10, path="eval/"):

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
                batch = batch.to(patch.device) / 255. # limit images to range [0-1]

                target_losses = []
                for position, target in zip(positions, targets):
                    scale_factor, tx, ty = position
                    noisy_transformations = gen_noisy_transformations(len(batch), scale_factor, tx, ty)
                    patch_batch = torch.cat([patch_t for _ in range(len(batch))])

                    mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations)
                    mod_img *= 255.  # convert input images back to range [0-255.]

                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patch.device)
                    # restrict patch+background to stay in range (0., 255.)
                    mod_img.clamp_(0., 255.)

                    # predict x, y, z, yaw
                    x, y, z, phi = model(mod_img)

                    # prepare shapes for MSE loss
                    #target_batch = target.repeat(len(batch), 1)
                    # TODO: improve readbility!
                    pred = torch.stack([x, y, z])
                    pred = pred.squeeze(2).mT

                    # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                    mask = torch.isnan(target)
                    target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)
                    target_batch = (pred * mask) + target
                    
                    target_losses.append(mse_loss(target_batch, pred))
                
                
                loss = torch.sum(torch.stack(target_losses))   
                actual_loss += loss.clone().detach()

                losses.append(loss.clone().detach())

                opt.zero_grad()
                loss.backward()
                opt.step()

                patch_t.data.clamp_(0., 1.)

                #optimized_patches.append(patch_t.clone().detach())
            actual_loss /= len(dataset)
            print("epoch {} loss {}".format(epoch, actual_loss))
            if actual_loss < best_loss:
                best_patch = patch_t.clone().detach()
                best_loss = actual_loss
        
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    #losses = torch.stack(losses)
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

            
            opt = torch.optim.Adam([scaling_factor, tx, ty], lr=lr)

            mask = torch.isnan(target)
            target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)
            
            for epoch in range(epochs):

                actual_loss = torch.tensor(0.).to(patch.device)
                for _, data in enumerate(dataset):
                    batch, _ = data
                    batch = batch.to(patch.device) / 255. # limit images to range [0-1]
            
                    noisy_transformations = gen_noisy_transformations(len(batch), scaling_factor, tx, ty)
                    patch_batch = torch.cat([patch for _ in range(len(batch))])
 
                    mod_img = place_patch(batch.clone(), patch_batch, noisy_transformations)
                    mod_img *= 255.  # convert input images back to range [0-255.]
                    # add noise to patch+background
                    mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(mod_img.shape).to(patch.device)    
                    mod_img.clamp_(0., 255.)

                    x, y, z, phi = model(mod_img)

                    # calculate mean l2 losses (target, prediction) for all images in batch
                    # prepare shapes for MSE loss
                    # TODO: improve readbility!
                    pred = torch.stack([x, y, z])
                    pred = pred.squeeze(2).mT

                    # only target x,y and z which were previously chosen, otherwise keep x/y/z to prediction
                    #torch.where(torch.isnan(target_batch), pred, target_batch)
                    target_batch = (pred * mask) + target

                    loss = mse_loss(target_batch, pred)

                    actual_loss += loss.clone().detach()

                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                actual_loss /= len(dataset)
                if actual_loss < best_loss:
                    print("restart {} improved loss to {}".format(restart, actual_loss))
                    best_loss = actual_loss
                    best_tx = tx.clone().detach()
                    best_ty = ty.clone().detach()
                    best_scaling = scaling_factor.clone().detach()

    except KeyboardInterrupt:
        print("Aborting optimization...")

    return best_scaling, best_tx, best_ty, best_loss

def calc_eval_loss(dataset, patch, transformation_matrix, model, target):
    actual_loss = torch.tensor(0.).to(patch.device)

    mask = torch.isnan(target)
    target = torch.where(mask, torch.tensor(0., dtype=torch.float32), target)

    for _, data in enumerate(dataset):
        batch, _ = data
        batch = batch.to(patch.device) / 255. # limit images to range [0-1]

        mod_img = place_patch(batch, patch, transformation_matrix)
        mod_img *= 255. # convert input images back to range [0-255.]
        mod_img.clamp_(0., 255.)

        x, y, z, phi = model(mod_img)

        # prepare shapes for MSE loss
        # target_batch = target.repeat(len(batch), 1)
        pred = torch.stack([x, y, z])
        pred = pred.squeeze(2).mT

        # only target x,y and z which are previously chosen, otherwise keep x/y/z to prediction
        #target_batch = torch.where(torch.isnan(target_batch), pred, target_batch)
        target_batch = (pred * mask) + target
        
        loss = mse_loss(target_batch, pred)
        actual_loss += loss.clone().detach()
    
    actual_loss /= len(dataset)

    return actual_loss



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


    # load the patch from misc folder
    if settings['patch']['mode'] == 'face':
        patch_start = np.load(settings['patch']['path'])
        patch_start = torch.from_numpy(patch_start).unsqueeze(0).unsqueeze(0).to(device) / 255.
        patch_start.clamp_(0., 1.)

    # or start from a random patch
    if settings['patch']['mode'] == 'random':
        patch_start = torch.rand(1, 1, 96, 160).to(device)

    # or start from a white patch
    if settings['patch']['mode'] == 'white':
        patch_start = torch.ones(1, 1, 300, 320).to(device)

    optimization_pos_losses = []
    optimization_pos_vectors = []

    optimization_patches = []
    optimization_patch_losses = []

    train_losses = []
    test_losses = []

    optimization_patches.append(patch_start)

    if mode == "split" or mode == "hybrid" or mode == "fixed":
        positions = torch.FloatTensor(len(targets), 3, 1).uniform_(-1., 1.).to(device)
    else:
        # start with placing the patch in the middle
        scale_factor, tx, ty = torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device), torch.tensor([0.0]).to(device)
        positions = []
        for target_idx in range(len(targets)):
            positions.append(torch.stack([scale_factor, tx, ty]))
        positions = torch.stack(positions)

    optimization_pos_vectors.append(positions)

    patch = patch_start.clone()
    
    for train_iteration in trange(num_hl_iter):
        
        pos_losses = []

        if mode == "split" or mode == "fixed":
            print("Optimizing patch...")
            patch, loss_patch = targeted_attack_patch(train_set, patch, model, optimization_pos_vectors[-1], targets=targets, lr=lr_patch, epochs=num_patch_epochs, path=path)
        elif mode == "joint" or mode == "hybrid":
            patch, loss_patch, positions = targeted_attack_joint(train_set, patch, model, optimization_pos_vectors[-1], targets=targets, lr=lr_patch, epochs=num_patch_epochs, path=path)
            optimization_pos_vectors.append(positions)

            pos_losses.append(loss_patch)

        optimization_patches.append(patch.clone())
        optimization_patch_losses.append(loss_patch)

        if mode == "split" or mode == "hybrid":
            # optimize positions for multiple target values
            positions = []
            for target_idx, target in enumerate(targets):
                print(f"Optimizing position for target {target.cpu().numpy()}...")
                scale_start, tx_start, ty_start = optimization_pos_vectors[-1][target_idx]
                scale_factor, tx, ty, loss_pos  = targeted_attack_position(train_set, patch, model, target, include_start=True, tx_start=tx_start, ty_start=ty_start, sf_start=scale_start, lr=lr_pos, num_restarts=num_pos_restarts, epochs=num_pos_epochs, path=path)
                positions.append(torch.stack([scale_factor, tx, ty]))
                pos_losses.append(loss_pos)
            positions = torch.stack(positions)
        elif mode == "fixed":
            pos_losses = [torch.tensor([0.])]

        optimization_pos_vectors.append(positions)
        print(optimization_pos_vectors[-1])
        optimization_pos_losses.append(torch.stack(pos_losses))

        train_loss = []
        test_loss = []
        for target_idx, target in enumerate(targets):
            scale_norm, tx_norm, ty_norm = norm_transformation(*optimization_pos_vectors[-1][target_idx])
            transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)

            train_loss.append(calc_eval_loss(train_set, patch, transformation_matrix, model, target))
            test_loss.append(calc_eval_loss(test_set, patch, transformation_matrix, model, target))

        train_losses.append(torch.stack(train_loss))    
        test_losses.append(torch.stack(test_loss))

    #print(optimization_patch_losses)
    optimization_patches = torch.stack(optimization_patches)
    optimization_patch_losses = torch.stack(optimization_patch_losses)
    
    #print(optimization_pos_vectors)
    optimization_pos_vectors = torch.stack(optimization_pos_vectors)
    optimization_pos_losses = torch.stack(optimization_pos_losses)
    

    train_losses = torch.stack(train_losses)
    test_losses = torch.stack(test_losses)

    print("Saving results...")
    # prepare data for plots
    # normalize scale factor, tx and ty for plots

    norm_optimized_vecs = [norm_transformation(optimization_pos_vectors[i].mT[..., 0], optimization_pos_vectors[i].mT[..., 1], optimization_pos_vectors[i].mT[..., 2]) for i in range(len(optimization_pos_vectors))]

    all_sf = torch.stack([norm_optimized_vecs[i][0] for i in range(len(norm_optimized_vecs))])
    all_tx = torch.stack([norm_optimized_vecs[i][1] for i in range(len(norm_optimized_vecs))])
    all_ty = torch.stack([norm_optimized_vecs[i][2] for i in range(len(norm_optimized_vecs))])


    # save all results in numpy arrays for later use
    np.save(path / 'patches.npy', optimization_patches.squeeze(1).squeeze(1).squeeze(1).cpu().numpy())
    np.save(path / 'patch_losses.npy', optimization_patch_losses.cpu().numpy())
    # np.save(path / 'positions.npy', optimization_pos_vectors.cpu().numpy())
    np.save(path / 'positions_norm.npy', np.array([all_sf.cpu().numpy(), all_tx.cpu().numpy(), all_ty.cpu().numpy()]))
    np.save(path / 'position_losses.npy', optimization_pos_losses.cpu().numpy())
    np.save(path / 'losses_train.npy', train_losses.cpu().numpy())
    np.save(path / 'losses_test.npy', test_losses.cpu().numpy())


    # final evaluation on test set
    print("Evaluation...")

    test_batch, test_gt = test_set.dataset[:]
    test_batch = test_batch.to(device) / 255. # limit images to range [0-1]

    boxplot_data = []
    #target_mask = torch.tensor(target_mask).to(patch.device)
    for target_idx, target in enumerate(targets):
        scale_norm, tx_norm, ty_norm = norm_transformation(*optimization_pos_vectors[-1][target_idx])
        transformation_matrix = get_transformation(scale_norm, tx_norm, ty_norm).to(device)
        pred_base = model(test_batch.float() * 255.)
        pred_base = torch.stack(pred_base[:3]).squeeze(2).mT
        target_batch = target.repeat(len(test_batch), 1)
        target_batch = torch.where(torch.isnan(target_batch), pred_base, target_batch)
        loss_base = torch.tensor([mse_loss(target_batch[i], pred_base[i]) for i in range(len(test_batch))])

        mod_img = place_patch(test_batch, patch_start, transformation_matrix)
        mod_img *= 255. # convert input images back to range [0-255.]
        mod_img.clamp_(0., 255.)
        pred_start_patch = model(mod_img.float())
        pred_start_patch = torch.stack(pred_start_patch[:3]).squeeze(2).mT
        target_batch = target.repeat(len(test_batch), 1)
        target_batch = torch.where(torch.isnan(target_batch), pred_start_patch, target_batch)
        loss_start_patch = torch.tensor([mse_loss(target_batch[i], pred_start_patch[i]) for i in range(len(test_batch))])

        mod_img = place_patch(test_batch, patch, transformation_matrix)
        mod_img *= 255. # convert input images back to range [0-255.]
        mod_img.clamp_(0., 255.)
        pred_opt_patch = model(mod_img.float())
        pred_opt_patch = torch.stack(pred_opt_patch[:3]).squeeze(2).mT
        target_batch = target.repeat(len(test_batch), 1)
        target_batch = torch.where(torch.isnan(target_batch), pred_opt_patch, target_batch)
        loss_opt_patch = torch.tensor([mse_loss(target_batch[i], pred_opt_patch[i]) for i in range(len(test_batch))])

        boxplot_data.append(torch.stack([loss_base.detach().cpu(), loss_start_patch.detach().cpu(), loss_opt_patch.detach().cpu()]))


    np.save(path / 'boxplot_data.npy', torch.stack(boxplot_data).cpu().numpy())

    # from plots import plot_results
    # plot_results(path)