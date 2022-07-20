import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

# def multi_image_attack(model, train_data, target_value, device, batch_size = 32, epochs=400, lr=1e-3, epsilon=5e-2):
#     for param in model.parameters():                   # freeze Frontnet completely
#         param.requires_grad = False 
    
#     target_x = torch.ones((batch_size, 1)).to(device) * target_value     # define target pose

#     in_dim = train_data.dataset.data[0].view(-1).shape[0]  # get the shape of one image from the dataset
#     perturbation_shape = in_dim                       # this will be the size of the first layer of the NN calculating the attack

#     adv_attack = PerturbationNN(in_dim, perturbation_shape).to(device)
#     #adv_attack = PerturbationModule(delta_shape, device)

#     opt = torch.optim.Adam(adv_attack.parameters())

#     #noise_jitter = transforms.Compose([
#     #                transforms.ColorJitter(),
#     #                transforms.GaussianBlur(kernel_size=(5,9))
#     #                ])

#     losses = []

#     try:
#         t = trange(epochs)
#         for i in t:
#             epoch_loss = []
#             for _, batch in enumerate(train_data):
#                 images, gt_position = batch
#                 images = images.to(device)


#                 # TODO: calculate single perturbation for whole batch instead of batch_size patches
#                 # the current code does not produce a reasonable attack!
#                 perturbations, angle, scale, shear = adv_attack(images)        # predict a new perturbation and parameters for affine transformation
#                                                                                # perturbation in range (0., 1.)
#                                                                                # angle in range (-180°, 180°)
#                                                                                # scale in range (0., 0.7)
#                                                                                # shear in range (-180°, 180°)

#                 patches = affine(perturbations, angle=float(angle[0]),         # apply affine transformation to the patch
#                            translate=[0, 0], 
#                            scale=float(scale[0]), 
#                            shear=float(shear[0])
#                            )

#                 parameters = [angle[0].detach().item(), 0,                         # store the optimized parameters in handy list
#                           0, scale[0].detach().item(), 
#                           shear[0].detach().item()]
            
#                 bit_mask = get_mask(perturbations, parameters).to(device)           # get the bit mask of the patch after affine transformation

#                 adv_images = ((images * bit_mask) + (patches*255.)) / 255.          # replace the pixels of the original image by the patch

#                 pred_x, pred_y, pred_z, pred_phi = model(adv_images)             # predict pose in adversarial image

#                 loss_x = torch.dist(pred_x, target_x, p=2)          # l2 distance between predicted and target pose
                
#                 loss = loss_x                       # other terms will later be added to the loss
#                 #loss_y = torch.dist(pred_y, target_y, p=2)
#                 #loss_z = torch.dist(pred_z, gt_z, p=2)
#                 #loss_phi = torch.dist(pred_phi, gt_phi, p=2)

#                 epoch_loss.append(loss.item())      # bookkeeping for later debugging 
#                 losses.append(loss.item())  

#                 t.set_postfix({"Avg. loss": np.mean(losses), "Avg. epoch loss": np.mean(epoch_loss)}, refresh=True)    # configure output of tqdm for monitoring         
#                 opt.zero_grad()                      # delete all accumulated gradients
#                 loss.backward()                      # calculate the gradient of the loss w.r.t the patch
#                 opt.step()                           # clip the pixel values of the adversarial image to stay in 0. - 255. 

#                 adv_images = torch.clamp(adv_images, 0., 255.)    # clip the pixel values of the adversarial image to stay in 0. - 255. 
                                                 
#     except KeyboardInterrupt:
#         print("Delta in range: ({},{})".format(torch.min(perturbation.data).item(), torch.max(perturbation.data).item()))
#         return perturbation             
#     print("Delta in range: ({},{})".format(torch.min(perturbation.data).item(), torch.max(perturbation.data).item()))
#     return perturbation


# def single_image_attack(model, image, target_value, device, iterations, lr=1e-4):
#     for param in model.parameters():                                    # freeze Frontnet completely
#         param.requires_grad = False 

#     delta_shape = image.shape

#     target_x = torch.ones((1, 1)).to(device) * target_value

#     #noise_jitter = transforms.Compose([
#     #                transforms.ColorJitter(),
#     #                transforms.GaussianBlur(kernel_size=(5,9))
#     #                ])

#     in_dim = image.view(-1).shape[0]
#     perturbation_shape = in_dim

#     adv_attack = PerturbationNN(in_dim, perturbation_shape, batch_size=1).to(device)#PerturbationModule(delta_shape, device)
#     opt = torch.optim.Adam(adv_attack.parameters())

#     try:
#         t = trange(iterations)
#         for i in t:
    
#             perturbation, angle, scale, shear = adv_attack(image)              # predict a new perturbation and parameters for affine transformation
#                                                                                # perturbation in range (0., 1.)
#                                                                                # angle in range (-180°, 180°)
#                                                                                # scale in range (0., 0.7)
#                                                                                # shear in range (-180°, 180°)
#             #print(angle, translate, scale, shear)
#             patch = affine(perturbation, angle=float(angle[0]),                # apply affine transformation to the patch
#                            translate=[0, 0], 
#                            scale=float(scale[0]), 
#                            shear=float(shear[0])
#                            )

#             parameters = [angle[0].detach().item(), 0,                         # store the optimized parameters in handy list
#                           0, scale[0].detach().item(), 
#                           shear[0].detach().item()]
            
#             bit_mask = get_mask(perturbation, parameters).to(device)           # get the bit mask of the patch after affine transformation

#             adv_image = ((image*bit_mask) + (patch*255.)) / 255.                # replace the pixels of the original image by the patch


#             #adv_image = adv_attack(image)
#             pred_x, pred_y, pred_z, pred_phi = model(adv_image)             # predict pose in adversarial image
#             #gt_x, gt_y, gt_z, gt_phi = model(image)


#             loss_x = torch.dist(pred_x, target_x, p=2)                    # l2 distance between predicted and target pose
#             #loss_y = torch.dist(pred_y, target_y, p=2)
#             #loss_z = torch.dist(pred_z, gt_z, p=2)
#             #loss_phi = torch.dist(pred_phi, gt_phi, p=2)

#             loss = loss_x                                               # other terms will later be added to the loss

#             t.set_postfix({"Loss": loss.item()}, refresh=True)          # configure output of tqdm for monitoring
#             if i % 1000 == 0:                                           # add further debug information after certain time step    
#                 print(pred_x)
#                 #print(adv_attack.affine_parameters.detach().cpu().numpy())
#                 print(angle[0].detach().item(), scale[0].detach().item(), 
#                       shear[0].detach().item())
#             opt.zero_grad()                                            # delete all accumulated gradients
#             loss.backward()                                            # calculate the gradient of the loss w.r.t the patch
#             opt.step()                                                 # update the pixel values of the patch

#             adv_image = torch.clamp(adv_image, 0., 255.)              # clip the pixel values of the adversarial image to stay in 0. - 255. 

#     except KeyboardInterrupt:
#         print("Stop calculating perturbation")

#     print("Delta in range: ({},{})".format(torch.min(perturbation.data).item(), torch.max(perturbation.data).item()))
#     gt_x, _, _, _ = model(image)
#     pred_x, _, _, _ = model(adv_image)         
#     print("Prediction: {}, ground-truth: {}".format(pred_x, gt_x))
#     print("Affine transformer parameters: ", angle[0].detach().item(), scale[0].detach().item(), 
#                                              shear[0].detach().item())
#     return perturbation


# class PerturbationNN(torch.nn.Module):
#     def __init__(self, in_dim, perturbation_shape, batch_size=32):
#         super(PerturbationNN, self).__init__()
#         self.batch_size = batch_size
#         self.linear1 = torch.nn.Linear(in_dim, 2500)
#         self.linear2 = torch.nn.Linear(2500, 2500)
#         self.perturbation_layer = torch.nn.Linear(2500, perturbation_shape)
#         #self.parameter_layer = torch.nn.Linear(2500, affinet_shape)
#         self.angle = torch.nn.Linear(2500, 1)
#         #self.translate = torch.nn.Linear(2500, 2)
#         self.scale = torch.nn.Linear(2500, 1)
#         self.shear = torch.nn.Linear(2500, 1)

#         self.activation = torch.nn.ReLU()


#     def forward(self, x):
#         img_shape = x.shape
#         x = x.view(self.batch_size, -1)
#         x = self.activation(self.linear1(x))
#         x = self.activation(self.linear2(x))
#         perturbation = self.perturbation_layer(x).view(img_shape)#torch.nn.functional.softmin(self.perturbation_layer(x).view(img_shape))
#         perturbation = (perturbation - torch.min(perturbation)) * 255. / (torch.max(perturbation) - torch.min(perturbation))
#         angle = torch.tanh(self.angle(x)) * 180.
#         #translate = self.translate(x)
#         scale = torch.nn.functional.softmin(self.scale(x))
#         scale = torch.clamp(scale, min=0., max=0.7)
#         shear = torch.tanh(self.shear(x)) * 180.


#         return perturbation, angle, scale, shear



# class PerturbationModule(torch.nn.Module):
#     def __init__(self, perturbation_shape, device):
#         super(PerturbationModule, self).__init__()
#         self.device = device
#         self.perturbation =  torch.nn.Parameter((torch.ones(perturbation_shape, device=self.device) * 255. ).requires_grad_(True))    # initialize perturbation
        
#         #angle, translate (List[int]), scale, shear
#         self.affine_parameters = torch.nn.Parameter(torch.tensor([0., 0.1, 0.3, 0.15, 180.], requires_grad=True, device=device))

#     def forward(self, x):
#         bit_mask = self.get_mask(self.perturbation, self.affine_parameters)
#         x *= bit_mask.to(self.device)
#         patch = affine(self.perturbation, angle=float(self.affine_parameters[0]), 
#                            translate=[int(self.affine_parameters[1]), int(self.affine_parameters[2])], 
#                            scale=float(self.affine_parameters[3]), 
#                            shear=float(self.affine_parameters[4])
#                            )

#         adv_img = (x + patch) / 255.

#         return adv_img

# def get_mask(perturbation, parameters):
#     #print(perturbation.shape)
#     #print(parameters)
#     # TODO: refactor for more efficient calculation
#     mask = affine(perturbation.detach().cpu(), angle=float(parameters[0]), 
#                         translate=[int(parameters[1]), int(parameters[2])], 
#                         scale=float(parameters[3]), 
#                         shear=float(parameters[4]),
#                         fill=-255.
#                         )


#     for row, column in enumerate(mask[0][0]):
#         for c_idx, value in enumerate(column):
#             if value != -255.:
#                 mask[0][0][row][c_idx] = 0.
#             else:
#                 mask[0][0][row][c_idx] = 1.
    
#     return mask




class Pose(torch.nn.Module):
    def __init__(self, device):
        super(Pose, self).__init__()
        self.device = device
        # generate initial random pose
        # random quaternions (source: http://planning.cs.uiuc.edu/node198.html)
        # 3 points u,v,w element of [0,1] uniformly at random
        u, v, w = torch.rand(3)
        # calculate quaternions
        qx = torch.sqrt(1-u) * torch.sin(2*torch.pi*v)
        qy = torch.sqrt(1-u)*torch.cos(2*torch.pi*v)
        qz = torch.sqrt(u)*torch.sin(2*torch.pi*w)
        qw = torch.sqrt(u)*torch.cos(2*torch.pi*w)
        self.quaternions = torch.stack([qx, qy, qz, qw])

        x = torch.rand(1) * 4.   # random number between 0. and 4.
        y = torch.rand(1) * 2.   # random number between 0. and 2.
        z = (-1 - 1) * torch.rand(1) + 1 # random value between -1. and 1.
        self.position = torch.concat([x, y, z])
        #print(self.position)

        self.pose = torch.nn.Parameter(torch.concat([self.position, self.quaternions]))
        print(self.pose)


if __name__=="__main__":
    from util import load_dataset, load_model
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    #pose = Pose(device)
    u, v, w = torch.rand(3)
    # calculate quaternions
    qx = torch.sqrt(1-u) * torch.sin(2*torch.pi*v)
    qy = torch.sqrt(1-u)*torch.cos(2*torch.pi*v)
    qz = torch.sqrt(u)*torch.sin(2*torch.pi*w)
    qw = torch.sqrt(u)*torch.cos(2*torch.pi*w)
    quaternions = torch.stack([qx, qy, qz, qw])

    x = torch.rand(1) * 4.   # random number between 0. and 4.
    y = torch.rand(1) * 2.   # random number between 0. and 2.
    z = (-1 - 1) * torch.rand(1) + 1 # random value between -1. and 1.
    position = torch.concat([x, y, z])
    #print(self.position)

    pose = (torch.concat([position, quaternions])).requires_grad_()
    print(pose)
    opt = torch.optim.Adam([pose])

    patch = (torch.rand(100, 100) * 255.)

    import yaml
    with open('adversarial_frontnet/camera_calibration/camera_config.yaml', 'r') as file:
        config = yaml.safe_load(file)



    
    # from patch_placement import place_patch

    # for i in range(1):
    #     image, _ = dataset.dataset.__getitem__(0)
    #     # print(image.shape)
    #     new_image = place_patch(image.squeeze(0), patch, pose, config)
    #     # import matplotlib.pyplot as plt
    #     # plt.imshow(new_image.detach().numpy())
    #     # plt.show()
    #     # print(new_image.shape)

    #     pred_pose_ori = torch.concat(model(image.unsqueeze(0)))
    #     pred_pose_patch = torch.concat(model(new_image.unsqueeze(0).unsqueeze(0)))

    #     #print(pred_pose_patch[0])
    #     target_x = 0.


    #     loss = - torch.linalg.norm(pred_pose_patch[0]-target_x, ord=2)

        

    #     opt.zero_grad()                                            # delete all accumulated gradients
    #     loss.backward()                                            # calculate the gradient of the loss w.r.t the patch
    #     opt.step() 

    #     print(i, loss.detach().numpy())
    #     print(pose)

    #image = dataset.dataset.data[70].unsqueeze(0).to(device)
    #patch = single_image_attack(model, image, 3., device, 1000000, lr=1e-4)

    # patches = multi_image_attack(model=model, train_data=dataset,
    #                                   target_value=3., device=device)

    # patches = patches.detach().cpu().numpy()
    # np.save('patches', patches)
