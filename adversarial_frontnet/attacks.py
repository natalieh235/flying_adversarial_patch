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

    target_x = torch.ones((1, 1)).to(device) * 3.

    #noise_jitter = transforms.Compose([
    #                transforms.ColorJitter(),
    #                transforms.GaussianBlur(kernel_size=(5,9))
    #                ])

    in_dim = image.view(-1).shape[0]
    perturbation_shape = in_dim
    parameters_shape = 5

    adv_attack = PerturbationNN(in_dim, perturbation_shape).to(device)#PerturbationModule(delta_shape, device)
    opt = torch.optim.Adam(adv_attack.parameters())

    try:
        t = trange(iterations)
        for i in t:
    
            perturbation, angle, translate, scale, shear = adv_attack(image)
            #print(angle, translate, scale, shear)
            patch = affine(perturbation, angle=float(angle[0]), 
                           translate=[int(0), int(0)], 
                           scale=float(scale[0]), 
                           shear=float(shear[0])
                           )

            parameters = [angle[0].detach().item(), 0, 
                          0, scale[0].detach().item(), 
                          shear[0].detach().item()]
            
            bit_mask = get_mask(perturbation, parameters).to(device)

            adv_image = ((image*bit_mask) + perturbation) / 255.


            #adv_image = adv_attack(image)
            pred_x, pred_y, pred_z, pred_phi = model(adv_image)             # Frontnet returns a list of tensors
            #gt_x, gt_y, gt_z, gt_phi = model(image)


            loss_x = torch.dist(pred_x, target_x, p=2)
            #loss_y = torch.dist(pred_y, target_y, p=2)
            #loss_z = torch.dist(pred_z, gt_z, p=2)
            #loss_phi = torch.dist(pred_phi, gt_phi, p=2)

            loss = loss_x

            t.set_postfix({"Loss": loss.item()}, refresh=True)
            if i % 1000 == 0:
                print(pred_x)
                #print(adv_attack.affine_parameters.detach().cpu().numpy())
                print(angle[0].detach().item(), translate[0][0].detach().item(), 
                      translate[0][1].detach().item(), scale[0].detach().item(), 
                      shear[0].detach().item())
            opt.zero_grad() 
            loss.backward()                                                                         # calculate the gradient w.r.t. the loss
            opt.step()    

            adv_image = torch.clamp(adv_image, 0., 255.)  

    except KeyboardInterrupt:
        print("Stop calculating perturbation")

    print("Delta in range: ({},{})".format(torch.min(perturbation.data).item(), torch.max(perturbation.data).item()))
    gt_x, _, _, _ = model(image)
    pred_x, _, _, _ = model(adv_image)         
    print("Prediction: {}, ground-truth: {}".format(pred_x, gt_x))
    print("Affine transformer parameters: ", angle[0].detach().item(), translate[0][0].detach().item(), 
                                             translate[0][1].detach().item(), scale[0].detach().item(), 
                                             shear[0].detach().item())
    return perturbation


class PerturbationNN(torch.nn.Module):
    def __init__(self, in_dim, perturbation_shape, batch_size=1):
        super(PerturbationNN, self).__init__()
        self.batch_size = batch_size
        self.linear1 = torch.nn.Linear(in_dim, 2500)
        self.linear2 = torch.nn.Linear(2500, 2500)
        self.perturbation_layer = torch.nn.Linear(2500, perturbation_shape)
        #self.parameter_layer = torch.nn.Linear(2500, affinet_shape)
        self.angle = torch.nn.Linear(2500, 1)
        self.translate = torch.nn.Linear(2500, 2)
        self.scale = torch.nn.Linear(2500, 1)
        self.shear = torch.nn.Linear(2500, 1)

        self.activation = torch.nn.ReLU()


    def forward(self, x):
        img_shape = x.shape
        x = x.view(self.batch_size, -1)
        x = self.activation(self.linear1(x))
        x = self.activation(self.linear2(x))
        perturbation = torch.nn.functional.softmin(self.perturbation_layer(x).view(img_shape)) *255.
        angle = torch.nn.functional.tanh(self.angle(x)) * 180.
        translate = self.translate(x)
        scale = torch.clamp(self.scale(x), 0.05, 0.5)
        shear = torch.nn.functional.tanh(self.shear(x)) * 180.


        return perturbation, angle, translate, scale, shear



class PerturbationModule(torch.nn.Module):
    def __init__(self, perturbation_shape, device):
        super(PerturbationModule, self).__init__()
        self.device = device
        self.perturbation =  torch.nn.Parameter((torch.ones(perturbation_shape, device=self.device) * 255. ).requires_grad_(True))    # initialize perturbation
        
        #angle, translate (List[int]), scale, shear
        self.affine_parameters = torch.nn.Parameter(torch.tensor([0., 0.1, 0.3, 0.15, 180.], requires_grad=True, device=device))

    def forward(self, x):
        bit_mask = self.get_mask(self.perturbation, self.affine_parameters)
        x *= bit_mask.to(self.device)
        patch = affine(self.perturbation, angle=float(self.affine_parameters[0]), 
                           translate=[int(self.affine_parameters[1]), int(self.affine_parameters[2])], 
                           scale=float(self.affine_parameters[3]), 
                           shear=float(self.affine_parameters[4])
                           )

        adv_img = (x + patch) / 255.

        return adv_img

def get_mask(perturbation, parameters):
    #print(perturbation.shape)
    #print(parameters)
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
    print(perturbation_shape)
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
    perturbation = ics(model, image, device, 1000000, lr=1e-7)

    perturbation = perturbation.detach().cpu().numpy()
    np.save('perturbation_x_3_w_parameters', perturbation)
