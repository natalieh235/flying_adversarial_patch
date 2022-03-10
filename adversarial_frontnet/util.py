import torch
import numpy as np
import sys

sys.path.insert(0,'../pulp-frontnet/PyTorch')
from Frontnet.Frontnet import FrontnetModel



def load_model(path, device, config):
    assert config in FrontnetModel.configs.keys(), 'config must be one of {}'.format(list(FrontnetModel.configs.keys()))
    
    model_params = FrontnetModel.configs[config]
    model = FrontnetModel(**model_params)
    
    try:
        model.load_state_dict(torch.load(path, map_location=device)['model'])
    except RuntimeError:
        print("RuntimeError while trying to load the saved model!")
        print("Seems like the model config does not match the saved model architecture.")
        print("Please check if you're loading the right model for the chosen config!")

    return model


if __name__=="__main__":
    path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = '160x32'
    model = load_model(path=path, device=device, config=config)