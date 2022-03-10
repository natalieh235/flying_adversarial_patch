from unittest import loader
import torch
import numpy as np
import sys

sys.path.insert(0,'../pulp-frontnet/PyTorch')
from Frontnet.Frontnet import FrontnetModel

from Frontnet.DataProcessor import DataProcessor
from Frontnet.Dataset import Dataset
from torch.utils import data

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
    model = FrontnetModel(**model_params)
    
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