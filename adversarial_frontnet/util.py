import torch
import sys

sys.path.insert(0,'/home/hanfeld/adversarial_frontnet/pulp-frontnet/PyTorch/')
from Frontnet.Frontnet import FrontnetModel

from Frontnet.DataProcessor import DataProcessor
from Frontnet.Dataset import Dataset
from torch.utils import data

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def load_model(path, device, config):
    """
    Loads a saved Frontnet model from the given path with the set configuration and moves it to CPU/GPU.
    Parameters
        ----------
        path
            The path to the stored Frontnet model
        device
            A PyTorch device (either CPU or GPU)
        config
            The architecture configuration of the Frontnet model. Must be one of ['160x32', '160x16', '80x32']
    """
    assert config in FrontnetModel.configs.keys(), 'config must be one of {}'.format(list(FrontnetModel.configs.keys()))
    
    # get correct architecture configuration
    model_params = FrontnetModel.configs[config]
    # initialize a random model with configuration
    model = FrontnetModel(**model_params).to(device)
    
    # load the saved model 
    try:
        model.load_state_dict(torch.load(path, map_location=device)['model'])
    except RuntimeError:
        print("RuntimeError while trying to load the saved model!")
        print("Seems like the model config does not match the saved model architecture.")
        print("Please check if you're loading the right model for the chosen config!")

    return model

def load_dataset(path, batch_size = 32, shuffle = False, drop_last = True, num_workers = 1):
    """
    Loads a dataset from the given path. 
    Parameters
        ----------
        path
            The path to the dataset
        batch_size
            The size of the batches the dataset will contain
        shuffle
            If set to True, the data will be shuffled randomly
        drop_last
            If set to True, the last batch of the dataset will be dropped. 
            This ensures that all returned batches are of the same size.
        num_workers
            Set the number of workers.
    """
    # load images and labels from the stored dataset
    [images, labels] = DataProcessor.ProcessTestData(path)
    # create a torch dataset from the loaded data
    dataset = Dataset(images, labels)

    # for quick and convinient access, create a torch DataLoader with the given parameters
    data_params = {'batch_size': batch_size, 'shuffle': shuffle, 'drop_last':drop_last, 'num_workers': num_workers}
    data_loader = data.DataLoader(dataset, **data_params)

    return data_loader


def calc_saliency(img, gt, model):
    input = img.unsqueeze(0).requires_grad_(True)
    prediction = torch.stack(model(input.float())).permute(1, 0, 2).squeeze(2).squeeze(0)

    loss_x = torch.nn.L1Loss()(prediction[0], gt[0])
    loss_y = torch.nn.L1Loss()(prediction[1], gt[1])
    loss_z = torch.nn.L1Loss()(prediction[2], gt[2])
    loss_phi = torch.nn.L1Loss()(prediction[3], gt[3])

    loss = loss_x + loss_y + loss_z + loss_phi

    loss.backward()

    saliency = input.grad.data.abs()

    return saliency

def plot_saliency(img, gt, model):
    saliency = calc_saliency(img, gt, model)
    img = img[0].detach().cpu().numpy()
    saliency = saliency[0][0].detach().cpu().numpy()

    fig, ax = plt.subplots(1, 3, figsize=(8, 2))
    ax[0].imshow(img, cmap='gray')
    ax[1].set_title('Saliency Map')
    ax[1].imshow(saliency, cmap='hot')
    ax[2].set_title('Superimposed')
    ax[2].imshow(img + (200000*saliency), cmap='gray')

    return fig

def plot_patch(patch, image, title='Plot', save=False, path='./'):

    img_min, img_max = patch.batch_place(image)

    f = plt.figure(constrained_layout=True, figsize=(10, 4))
    subfigs = f.subfigures(1, 2, width_ratios=[1, 3])
    fig_patch = subfigs[0].subplots(1,1)
    fig_patch.imshow(patch.patch[0][0].detach().cpu().numpy(), cmap='gray')
    subfigs[0].suptitle('Patch', fontsize='x-large')

    subfigs[1].suptitle('placed', fontsize='x-large')
    fig_placed = subfigs[1].subplots(1,2)
    fig_placed[0].imshow(img_min[0][0].detach().cpu().numpy(), cmap='gray')
    fig_placed[0].set_title('min direction')
    fig_placed[1].imshow(img_max[0][0].detach().cpu().numpy(), cmap='gray')
    fig_placed[1].set_title('max direction')

    f.suptitle(title, fontsize='xx-large')
    
    if save:
        plt.savefig(path+title+'.jpg', transparent=False)
        plt.close()
    else: 
        return f


