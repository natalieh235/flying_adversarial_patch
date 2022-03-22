from json import load
import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms

def ucs(model, train_data, delta_shape, device, batch_size = 32, epochs=400, lr=1e-3, epsilon=5e-2):
    """
    Universal, class specific attack
    """
    #target_x = target_y = target_z = target_phi = torch.zeros((batch_size,1), device=device)
    #target_z = torch.ones((batch_size,1), device=device) * -5
    target_x = torch.ones((batch_size, 1)).to(device) * 3.
    
    delta = torch.zeros(delta_shape, requires_grad=True, device=device)                       # initialize perturbation
    affine_transformer = transforms.RandomAffine(degrees=(-70, 70), translate=(0.1, 0.3), scale=(0.1, 0.6))
    #delta = torch.randint(low=-1000, high=1000, size=delta_shape, device=device) / 1000.
    #delta.requires_grad_(True)

    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam([delta], lr=lr)                                                      # and optimizer
    losses = []

    try:
        t = trange(epochs)
        for i in t:
            epoch_loss = []
            for _, batch in enumerate(train_data):
                images, gt_position = batch 
                images = images.to(device)
                #adv_example = images + delta                                                
                # pred_x, pred_y, pred_z, pred_phi = model(adv_example) 

                #transformed_batch = [affine_transfomer(delta) for _ in range(batch_size)]
                #
                #adv_batch = images.clone()
                #for i, image in enumerate(images):
                #    transformed_delta = affine_transformer(delta)
                #    adv_batch[i] = torch.add(image, transformed_delta)

                adv_batch = torch.add(images, delta)
                adv_batch = torch.clamp(adv_batch, 0., 1.)                  
                
                pred_x, _, _, _ = model(adv_batch)             # Frontnet returns a list of tensors
                #pred = torch.stack(pred)                      # we therefore concatenate the tensors in the list
                #pred = pred.view(gt_position.shape)
                #print(pred_x.shape)

                #gt_x, _, _, _ = model(images)
                #print(gt_x.shape)
                #print(gt_x[0])
                #gt_position = model(images)
                #gt_position = torch.stack(gt_position).view(pred.shape)
                #pred = pred.T                               # and transpose the new tensor to match the shape of the stored position



                #loss = loss_x + loss_y + loss_z + loss_phi
                #loss = loss_z
                #loss = -criterion(pred, gt_position)
                loss = torch.dist(pred_x, target_x, p=2)
                epoch_loss.append(loss.item())
                losses.append(loss.item())

                t.set_postfix({"Avg. loss": np.mean(losses), "Avg. epoch loss": np.mean(epoch_loss)}, refresh=True)            
                opt.zero_grad() 
                loss.backward()                                                                         # calculate the gradient w.r.t. the loss
                opt.step()                                                                              # update the perturbation

                #adv_batch = torch.clamp(adv_batch, 0., 1.)   

                #delta = torch.clamp(delta, -epsilon, epsilon)                                                    # clip the pixel values to stay in certain range
    except KeyboardInterrupt:
        print("Delta in range: ({},{})".format(torch.min(delta.data).item(), torch.max(delta.data).item()))
        return delta             
    print("Delta in range: ({},{})".format(torch.min(delta.data).item(), torch.max(delta.data).item()))
    return delta



def attack_momentum(model, dataset, device, epochs=5, epsilon=1.0):

    delta_shape =  dataset.dataset.data[0].shape
    print(delta_shape)
    #delta = torch.zeros(delta_shape, device=device)                       # initialize perturbation
    delta = torch.randint(low=-1000, high=1000, size=delta_shape, device=device) / 1000.
    delta.requires_grad_(True)
    
    alpha = 1.
    g_t = torch.zeros(next(iter(dataset))[0].shape)
    print(g_t.shape)

    criterion = torch.nn.L1Loss()

    for i in range(epochs):
        epoch_loss = []
        for _, batch in enumerate(tqdm(dataset)):
            images, gt_position = batch
            
            images.requires_grad_()
            delta.requires_grad_()
            adv_batch = torch.add(images, delta)

            pred = model(adv_batch)                         # Frontnet returns a list of tensors
            pred = torch.stack(pred)                      # we therefore concatenate the tensors in the list
            pred = pred.view(gt_position.shape)

            gt_position = model(images)
            gt_position = torch.stack(gt_position).view(pred.shape)

            loss = - criterion(pred, gt_position)        # maximize the distance between the prediction and the stored true position
            epoch_loss.append(loss.item())
            
            grad = torch.autograd.grad(loss, adv_batch,
                                        retain_graph=False, create_graph=False)[0]

            grad = grad / torch.norm(grad,p=1)
            grad = grad + g_t*1.0
            g_t = grad


            adv_batch = adv_batch.detach() + alpha*grad / torch.norm(grad,float('inf'))
            adv_batch = torch.clamp(adv_batch, min=0., max=1.)

        print("Avg. epoch loss: %.2f" % (np.mean(epoch_loss)))
    
    return delta
    #delta_image = delta.detach().cpu().squeeze(0).permute(1,2,0).numpy()
    #delta_image = Image.fromarray(delta_image)
    #delta_image.save("perturbation.png")


if __name__=="__main__":
    from util import load_dataset, load_model
    torch.multiprocessing.set_sharing_strategy('file_system')
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    #attack_momentum(model, dataset, device)
 
    perturbation_shape = dataset.dataset.data[0].shape
    perturbation = ucs(model, dataset, perturbation_shape, device, epochs=1000, batch_size=32, lr=1e-4)
    #perturbation = attack_momentum(model, dataset, device)
    perturbation = perturbation.detach().cpu().numpy()
    np.save('perturbation', perturbation)
