from json import load
import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

def ucs(model, train_data, delta_shape, target_value, device, batch_size = 32, epochs=400, lr=1e-3, epsilon=5e-2):
    """
    Universal, class specific attack
    """
    #target_x = target_y = target_z = target_phi = torch.zeros((batch_size,1), device=device)
    #target_z = torch.ones((batch_size,1), device=device) * -5
    target_x = torch.ones((batch_size, 1)).to(device) * target_value
    
    delta = torch.zeros(delta_shape, requires_grad=True, device=device)                       # initialize perturbation
    affine_transformer = transforms.RandomAffine(degrees=(-70, 70), translate=(0.1, 0.3), scale=(0.1, 0.6))
    #delta = torch.randint(low=-1000, high=1000, size=delta_shape, device=device) / 1000.
    #delta.requires_grad_(True)

    #criterion = torch.nn.L1Loss()
    #criterion = torch.nn.MSELoss()
    opt = torch.optim.Adam([delta], lr=lr) # parameter für affine transformation mit trainieren                                                     # and optimizer
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

                patch = affine_transformer(delta)

                adv_batch = torch.add(images, patch)#delta)
                #adv_batch = torch.clamp(adv_batch, 0., 1.)                  
                
                # use tanh or sigmoid instead


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
                loss_positon = torch.dist(pred_x, target_x, p=2)
                
                loss = loss_positon 
                
                epoch_loss.append(loss.item())
                losses.append(loss.item())

                t.set_postfix({"Avg. loss": np.mean(losses), "Avg. epoch loss": np.mean(epoch_loss)}, refresh=True)            
                opt.zero_grad() 
                loss.backward()                                                                         # calculate the gradient w.r.t. the loss
                opt.step()                                                                              # update the perturbation

                adv_batch = torch.clamp(adv_batch, 0., 255.)   
                #delta.data.clamp_(-1., 1.)
                #delta = torch.clamp(delta, -1., 1.)                                                    # clip the pixel values to stay in certain range
    except KeyboardInterrupt:
        print("Delta in range: ({},{})".format(torch.min(delta.data).item(), torch.max(delta.data).item()))
        return delta             
    print("Delta in range: ({},{})".format(torch.min(delta.data).item(), torch.max(delta.data).item()))
    return delta


def ics(model, image, device, iterations, lr=1e-4):
    for param in model.parameters():
        param.requires_grad = False 

    delta_shape = image.shape

    #image = image.to(device) / torch.max(image) #*255.
    #print(torch.min(image), torch.max(image))
    
    target_x = torch.ones((1, 1)).to(device) * 3.
    
    #affine_transformer = transforms.RandomAffine(degrees=(-70, 70), translate=(0.1, 0.3), scale=(0.1, 0.6))

    delta = torch.ones(delta_shape, device=device) * 255.                   # initialize perturbation
    #delta = torch.rand(delta_shape, device=device) * 10.
    delta.requires_grad_(True)

    affine_parameters = torch.tensor([20., 0.1, 0.3, 0.2, 180.], requires_grad=True, device=device)
    affine_parameters.requires_grad_(True)

    noise_jitter = transforms.Compose([
                    transforms.ColorJitter(),
                    transforms.GaussianBlur(kernel_size=(5,9))
                    ])

    opt = torch.optim.Adam([delta, affine_parameters], lr=lr)

    try:
        t = trange(iterations)
        for i in t:
            #gt_x, _, _, _ = model(image)
            #patch = affine_transformer(delta)
            patch = affine(delta, angle=float(affine_parameters[0]), 
                           translate=[int(affine_parameters[1]), int(affine_parameters[2])], 
                           scale=float(affine_parameters[3]), 
                           shear=float(affine_parameters[4])
                           )

            mask = get_mask(delta, affine_parameters).to(device)  # get a mask of the placement of the patch in the image
            image_p = image * mask                              # delete the pixel values in the original image where the mask is


            adv_image = image_p + noise_jitter(patch)
            pred_x, pred_y, pred_z, pred_phi = model(adv_image)             # Frontnet returns a list of tensors
            #gt_x, gt_y, gt_z, gt_phi = model(image)


            loss_x = torch.dist(pred_x, target_x, p=2)
            #loss_y = torch.dist(pred_y, target_y, p=2)
            #loss_z = torch.dist(pred_z, gt_z, p=2)
            #loss_phi = torch.dist(pred_phi, gt_phi, p=2)
            
            #loss_similarity = torch.dist((image+delta), image, p=2)

            loss = loss_x

            t.set_postfix({"Loss": loss.item()}, refresh=True)
            if i % 1000 == 0:
                print(pred_x)
                print(affine_parameters.detach().cpu().numpy())

            opt.zero_grad() 
            loss.backward()                                                                         # calculate the gradient w.r.t. the loss
            opt.step()    

            adv_image = torch.clamp(adv_image, 0., 255.)  

            #delta.data.clamp_(-0.3, 0.3)
            #delta = torch.nn.Tanh()(delta)
    except KeyboardInterrupt:
        print("Stop calculating perturbation")

    print("Delta in range: ({},{})".format(torch.min(delta.data).item(), torch.max(delta.data).item()))
    gt_x, _, _, _ = model(image)
    pred_x, _, _, _ = model(image + delta)             # Frontnet returns a list of tensors
    print("Prediction: {}, ground-truth: {}".format(pred_x, gt_x))
    print("Affine transformer parameters: ", affine_parameters.detach().cpu().numpy())
    return delta


def get_mask(perturbation, parameters):
    mask = affine(perturbation.detach().cpu(), angle=float(parameters[0]), 
                           translate=[int(parameters[1]), int(parameters[2])], 
                           scale=float(parameters[3]), 
                           shear=float(parameters[4]),
                           fill=-255.
                           )

    for row, column in enumerate(mask[0][0]):
        for c_idx, value in enumerate(column):
            if value != -255.:
                mask[0][0][row][c_idx] = 0.
            else:
                mask[0][0][row][c_idx] = 1.
    
    return mask



if __name__=="__main__":
    from util import load_dataset, load_model
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    #attack_momentum(model, dataset, device)
 
    perturbation_shape = dataset.dataset.data[0].shape

    #print(dataset.dataset)
    #perturbation = ucs(model, dataset, perturbation_shape, device, epochs=1000000, batch_size=32, lr=1e-4)
    #perturbation = attack_momentum(model, dataset, device)

    # for i, img in enumerate(dataset.dataset.data):
    #     img = img.to(device).unsqueeze(0)
    #     gt_x, _, _, _ = model(img)
    #     if int(gt_x) == 1:
    #         print(i, gt_x)
    #         image = img
    #         break
    image = dataset.dataset.data[70].unsqueeze(0).to(device)
    perturbation = ics(model, image, device, 1000000, lr=3e-2)

    perturbation = perturbation.detach().cpu().numpy()
    np.save('perturbation_x_3_w_parameters', perturbation)
