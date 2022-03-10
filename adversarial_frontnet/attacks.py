from json import load
import numpy as np
import torch
from tqdm import trange

def ucs(model, train_data, delta_shape, device, batch_size = 32, epochs=400, lr=1e-3, epsilon=5e-2):
    """
    Universal, class specific attack
    """
    #target_x = target_y = target_z = target_phi = torch.zeros((batch_size,1), device=device)
    target_z = torch.ones((batch_size,1), device=device) * -5
    
    delta = torch.zeros(delta_shape, requires_grad=True, device=device)                       # initialize perturbation
    
    #criterion = torch.nn.L1Loss()
    criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam([delta], lr=lr)                                                      # and optimizer
    losses = []

    t = trange(epochs)
    for i in t:
        epoch_loss = []
        for _, batch in enumerate(train_data):
            images, _ = batch 
            adv_example = images + delta                                                
            pred_x, pred_y, pred_z, pred_phi = model(adv_example)                   
            
            #loss_x = criterion(pred_x, target_x)
            #loss_y = criterion(pred_y, target_y)
            loss_z = criterion(pred_z, target_z)
            #loss_phi = criterion(pred_phi, target_phi)

            #loss = loss_x + loss_y + loss_z + loss_phi
            loss = loss_z
            epoch_loss.append(loss.item())
            losses.append(loss.item())

            t.set_postfix({"Avg. loss": np.mean(losses), "Avg. epoch loss": np.mean(epoch_loss)}, refresh=True)            
            opt.zero_grad() 
            loss.backward()                                                                         # calculate the gradient w.r.t. the loss
            opt.step()                                                                              # update the perturbation

            
            delta.data.clamp_(-epsilon, epsilon)                                                    # clip the pixel values to stay in certain range
                 
    print("Delta in range: ({},{})".format(torch.min(delta.data).item(), torch.max(delta.data).item()))
    return delta

if __name__=="__main__":
    from util import load_dataset, load_model

    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    dataset = load_dataset(path=dataset_path)

    perturbation_shape = dataset.dataset.data[0].shape
    perturbation = ucs(model, dataset, perturbation_shape, device)
