import torch
import numpy as np
import sys
import os

import matplotlib.pyplot as plt

sys.path.insert(0,'../pulp-frontnet/PyTorch')
sys.path.insert(0,'../adversarial_frontnet/')
from Frontnet.Frontnet import FrontnetModel

from adversarial_frontnet.util import load_dataset, load_model

from adversarial_frontnet.patch_placement import place_patch

from tqdm import tqdm


def plots_tx_ty(base_img, patches, path):
    print("Plotting tx / ty values for fixed ty / tx ...")
    for num, patch in enumerate(tqdm(patches)):

        ## create folder
        os.makedirs(path+f'patch{num}/', exist_ok=True)

        ## loop tx
        ty = 0.

        all_y = []

        for tx in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, tx], [0, 0.4, ty]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(base_img, patch, transformation_matrix)


            prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
            all_y.append(prediction_mod.detach()[1].item())
        
        ## loop ty
        tx = 0.

        all_z = []

        for ty in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, tx], [0, 0.4, ty]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(base_img, patch, transformation_matrix)


            prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
            all_z.append(prediction_mod.detach()[2].item())

        plt.plot(np.linspace(-1, 1, num=201), all_y)
        plt.grid('True')
        plt.xlabel('tx')
        plt.ylabel('y')
        plt.savefig(path+f'patch{num}/'+'tx_only.jpg', dpi=200)
        plt.close()

        plt.plot(np.linspace(-1, 1, num=201), all_z)
        plt.grid('True')
        plt.xlabel('ty')
        plt.ylabel('z')
        plt.savefig(path+f'patch{num}/'+'ty_only.jpg', dpi=200)
        plt.close()


def save_plots_noise(values, name, xlabel, ylabel):
        plt.plot(np.linspace(-1, 1, num=201), np.array(values)[..., :5], label= ['σ = 0', 'σ = 10', 'σ = 20', 'σ = 30', 'σ = 40'])
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid('True')
        plt.savefig(name+'-40.jpg', dpi=200)
        plt.close()

        plt.plot(np.linspace(-1, 1, num=201), np.array(values)[..., 0], label='σ = 0')
        plt.plot(np.linspace(-1, 1, num=201), np.array(values)[..., 5:], label=['σ = 50', 'σ = 60', 'σ = 70', 'σ = 80', 'σ = 90', 'σ = 100'])
        plt.legend()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid('True')
        plt.savefig(name+'50-100.jpg', dpi=200)
        plt.close()

def noise_analysis_single(base_img, patches, path):
    print('Plotting analysis for different noise intensities...')
    batch_clear_img = []
    for i in range(11):
        batch_clear_img.append(base_img[0])

    batch_clear_img = torch.stack(batch_clear_img)

    ## noise only on base image
    noisy_batch = batch_clear_img.clone()

    for i in np.linspace(10, 100, num=10):
        noise = torch.distributions.normal.Normal(loc=0.0, scale=i).sample(batch_clear_img[0].shape)   #loc == mu, scale == sigma
        noisy_batch[int((i-10)/10)+1] += noise
        noisy_batch[int((i-10)/10)+1].clamp_(0., 255.)

    for num, patch in enumerate(tqdm(patches)):

        ## create folder
        os.makedirs(path+f'patch{num}/', exist_ok=True)

        ty = 0.

        all_y_noisy = []

        for tx in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, tx], [0, 0.4, ty]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(noisy_batch.clone(), patch, transformation_matrix)

            prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
            all_y_noisy.append(prediction_mod[..., 1].detach().clone().numpy())

        tx = 0.

        all_z_noisy = []

        for ty in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, tx], [0, 0.4, ty]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(noisy_batch.clone(), patch, transformation_matrix)

            prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
            all_z_noisy.append(prediction_mod[..., 2].detach().clone().numpy())


        save_plots_noise(all_y_noisy, path+f'patch{num}/'+'noisy_base_tx_only', 'tx', 'y')

        save_plots_noise(all_z_noisy, path+f'patch{num}/'+'noisy_base_ty_only', 'ty', 'z')


        ## noise on base+patch
        ty = 0.

        all_y_noisy_patch = []

        for tx in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, tx], [0, 0.4, ty]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(batch_clear_img.clone(), patch, transformation_matrix)
            
            noisy_batch = mod_img.clone()

            for i in np.linspace(10, 100, num=10):
                noise = torch.distributions.normal.Normal(loc=0.0, scale=i).sample(batch_clear_img[0].shape)   #loc == mu, scale == sigma
                noisy_batch[int((i-10)/10)+1] += noise
                noisy_batch[int((i-10)/10)+1].clamp_(0., 255.)


            prediction_mod = torch.stack(model(noisy_batch)).permute(1, 0, 2).squeeze(2).squeeze(0)
            all_y_noisy_patch.append(prediction_mod[..., 1].detach().clone().numpy())

        tx = 0.

        all_z_noisy_patch = []

        for ty in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, tx], [0, 0.4, ty]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(batch_clear_img.clone(), patch, transformation_matrix)
            
            noisy_batch = mod_img.clone()

            for i in np.linspace(10, 100, num=10):
                noise = torch.distributions.normal.Normal(loc=0.0, scale=i).sample(batch_clear_img[0].shape)   #loc == mu, scale == sigma
                noisy_batch[int((i-10)/10)+1] += noise
                noisy_batch[int((i-10)/10)+1].clamp_(0., 255.)


            prediction_mod = torch.stack(model(noisy_batch)).permute(1, 0, 2).squeeze(2).squeeze(0)
            all_z_noisy_patch.append(prediction_mod[..., 2].detach().clone().numpy())

        save_plots_noise(all_y_noisy_patch, path+f'patch{num}/'+'noisy_patch_tx_only', 'tx', 'y')

        save_plots_noise(all_z_noisy_patch, path+f'patch{num}/'+'noisy_patch_ty_only', 'ty', 'z')


def noise_analysis_batch(base_img, patches, batch_size=100):
    print(f'Plotting average y and z values for batch of {batch_size} base images with random noise...')
    batch_clear_img = []
    for i in range(batch_size):
        batch_clear_img.append(base_img[0])

    batch_clear_img = torch.stack(batch_clear_img)

    for num, patch in enumerate(tqdm(patches)):

        ## create folder
        os.makedirs(path+f'patch{num}/', exist_ok=True)
        

        y_random_noisy = []
        z_random_noisy = []

        for t in np.linspace(-1, 1, num=201):
            transformation_matrix = [[[0.4, 0, t], [0, 0.4, 0]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(batch_clear_img.clone(), patch, transformation_matrix)
            mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch_clear_img.shape)
            mod_img.clamp_(0., 255.)


            prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
            y_random_noisy.append(prediction_mod[..., 1].detach().clone().numpy())

            transformation_matrix = [[[0.4, 0, 0], [0, 0.4, t]]]
            transformation_matrix = torch.tensor(transformation_matrix).float()
            mod_img = place_patch(batch_clear_img.clone(), patch, transformation_matrix)
            mod_img += torch.distributions.normal.Normal(loc=0.0, scale=10.).sample(batch_clear_img.shape)
            mod_img.clamp_(0., 255.)

            prediction_mod = torch.stack(model(mod_img.float())).permute(1, 0, 2).squeeze(2).squeeze(0)
            z_random_noisy.append(prediction_mod[..., 2].detach().clone().numpy())

        plt.plot(np.linspace(-1, 1, num=len(y_random_noisy)), np.array(y_random_noisy), color='lightsteelblue')
        plt.plot(np.linspace(-1, 1, num=len(y_random_noisy)), np.mean(np.array(y_random_noisy), axis=1))
        plt.title(f'Mean over {batch_size} noisy images')
        plt.xlabel('tx')
        plt.ylabel('y')
        plt.grid('True')
        plt.savefig(path+f'patch{num}/'+'noisy_batch_tx.jpg', dpi=200)
        plt.close()

        plt.plot(np.linspace(-1, 1, num=len(y_random_noisy)), np.array(z_random_noisy), color='lightsteelblue')
        plt.plot(np.linspace(-1, 1, num=len(y_random_noisy)), np.mean(np.array(z_random_noisy), axis=1))
        plt.title(f'Mean over {batch_size} noisy images')
        plt.xlabel('ty')
        plt.ylabel('z')
        plt.grid('True')
        plt.savefig(path+f'patch{num}/'+'noisy_batch_ty.jpg', dpi=200)
        plt.close()



if __name__ == '__main__':
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()


    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=False, drop_last=True, num_workers=0)

    # custom_patch_1 = np.load("custom_patch_resized.npy")
    custom_patch_2 = np.load("misc/custom_patches/custom_patch2_resized.npy")
    custom_patch_3 = np.load("misc/custom_patches/custom_patch3_resized.npy")
    custom_patch_4 = np.load("misc/custom_patches/custom_patch4_resized.npy")
    custom_patch_5 = np.load("misc/custom_patches/custom_patch5_resized.npy")

    patches = [torch.tensor(custom_patch_2).unsqueeze(0).unsqueeze(0).to(device), torch.tensor(custom_patch_3).unsqueeze(0).unsqueeze(0).to(device), torch.tensor(custom_patch_4).unsqueeze(0).unsqueeze(0).to(device), torch.tensor(custom_patch_5).unsqueeze(0).unsqueeze(0).to(device)]
   
    
    path = "eval/custom_patches_eval/white_base/"
    os.makedirs(path, exist_ok=True)
    
    # perform tests on white image
    white_img = (torch.ones(1, 1, 96, 160) * 255.).to(device)

    plots_tx_ty(white_img, patches, path)
    noise_analysis_single(white_img, patches, path)
    noise_analysis_batch(white_img, patches)


