import numpy as np
import torch
from tqdm import tqdm, trange

from torchvision import transforms
from torchvision.transforms.functional import affine

from patch_placement import place_patch

class Patch(torch.nn.Module):
    def __init__(self, patch_size=[1, 50, 50]):
        super(Patch, self).__init__()
        patch = (torch.rand(1, *patch_size) * 255.)#.requires_grad_()
        self.patch = torch.nn.Parameter(patch)

        transformation_min = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]])
        transformation_max = torch.tensor([[[*torch.rand(3,)],[*torch.rand(3,)]]])

        self.transformation_min = torch.nn.Parameter(transformation_min)
        self.transformation_max = torch.nn.Parameter(transformation_max)

    def place(self, image):
        image_min = place_patch(image, self.patch, self.transformation_min)
        image_max = place_patch(image, self.patch, self.transformation_max)
        return torch.stack([image_min, image_max])

    def batch_place(self, batch):
        out = torch.stack([self.place(img) for img in batch]).view((2, *batch.shape))
        batch_min, batch_max = out

        return batch_min, batch_max

class TargetedAttack():
    def __init__(self, model, dataset, target=[True, False, False, False], learning_rate = 1e-3, path='eval/targeted/'):
        self.model = model
        self.dataset = dataset
        
        self.path = path

        # define the maximum and minimum targets to optimize towards
        self.target = torch.tensor(target)

        # init random patch
        self.x_patch = Patch()

        self.optimizer = torch.optim.Adam(self.x_patch.parameters(), lr = learning_rate)

    def optimize(self, epochs=100000):
        try:
            losses = []
            for i in range(epochs):
                for _ in range(len(self.dataset)):
                    image_batch, pose_batch = next(iter(self.dataset))
                    # place patch with current parameters in input image
                    input_images_min, input_images_max = self.x_patch.batch_place(image_batch)

                    targets_min = pose_batch * ~self.target
                    targets_max = pose_batch * self.target + (4*self.target.int())

                    # get prediction of current pose from NN
                    pred_attack_min = torch.stack(self.model(input_images_min))
                    pred_attack_min = pred_attack_min.view(dataset.batch_size, -1)  # get prediction into appropriate shape
                    
                    pred_attack_max = torch.stack(self.model(input_images_max))
                    pred_attack_max = pred_attack_max.view(dataset.batch_size, -1)  # get prediction into appropriate shape

                    # calculate loss between target pose and predicted pose
                    loss_min = torch.dist(targets_min, pred_attack_min, p=2)
                    loss_max = torch.dist(targets_max, pred_attack_max, p=2)

                    # average loss between the two loss terms
                    loss = (loss_min + loss_max) / 2.

                    # save loss for evaluation
                    losses.append(loss.detach().numpy())

                    # perform update step
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    # restrict patch pixel values to stay between 0 and 255
                    self.x_patch.patch.data.clamp_(0., 255.)

                # occasional print for observering training process
                if i % 1 == 0:
                    print("step %d, loss %.6f"  % (i, loss))

                    print("Min: ", self.x_patch.patch_parameters_min.detach().numpy())
                    print("Max: ", self.x_patch.patch_parameters_max.detach().numpy())
            
        except KeyboardInterrupt:                   # training process can be interrupted anytime
            print("Aborting optimization...")    

        print("Bing!")
        print("Last loss: ", loss.detach().numpy())
        print("Last prediciton: ", pred_attack_min, pred_attack_max)

        np.save(self.path+'losses_test', losses)

        return self.x_patch

def targeted_attack(image, target, model, path="eval/targeted/"):
    # initialize optimizer
    x_patch = Patch()
    opt = torch.optim.Adam(x_patch.parameters(), lr=3e-5)
    prediction = torch.concat(model(image)).squeeze(1)
    
    new_image_min = place_patch(image, x_patch.patch, x_patch.transformation_min)
    new_image_max = place_patch(image, x_patch.patch, x_patch.transformation_max)
    
    pred_attack_min = torch.concat(model(new_image_min)).squeeze(1)
    pred_attack_max = torch.concat(model(new_image_max)).squeeze(1)
    loss_min = torch.dist(prediction, pred_attack_min, p=2)
    loss_max = torch.dist(prediction, pred_attack_max, p=2)

    loss = (loss_min + loss_max) / 2.

    losses = []
    losses.append(loss.detach().numpy())

    target_min = torch.tensor([0., *prediction[1:]])
    target_max = torch.tensor([4., *prediction[1:]])

    i = 0.
    try:
        while loss > 0.01:
            i += 1
            # prediction = torch.concat(model(new_image)).squeeze(1)
            # loss = torch.dist(prediction, target, p=2)
            pred_attack_min = torch.concat(model(new_image_min)).squeeze(1)
            pred_attack_max = torch.concat(model(new_image_max)).squeeze(1)
            loss_min = torch.dist(pred_attack_min, target_min, p=2)
            loss_max = torch.dist(pred_attack_max, target_max, p=2)

            loss = (loss_min + loss_max) / 2.
            losses.append(loss.detach().numpy())
            opt.zero_grad()
            loss.backward()
            opt.step()

            x_patch.patch.data.clamp_(0., 255.)
            # new_image = place_patch(image, x_patch.patch, x_patch.transformation_min)
            new_image_min = place_patch(image, x_patch.patch, x_patch.transformation_min)
            new_image_max = place_patch(image, x_patch.patch, x_patch.transformation_max)
    
            if i % 100 == 0:
                print("step %d, loss %.6f" % (i, loss))
                print("transformation matrix min: ", x_patch.transformation_min.detach().numpy())
                print("transformation matrix max: ", x_patch.transformation_max.detach().numpy())
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().numpy())
    print("Last prediciton: ", pred_attack_min, pred_attack_max)

    np.save(path+'losses_test', losses)

    return x_patch.patch, x_patch.transformation_min


def untargeted_attack(image, patch, model, transformation_matrix, path='eval/untargeted/'): # angle, scale, tx, ty,
    # initialize optimizer
    opt = torch.optim.Adam([transformation_matrix], lr=1e-4)
    prediction = torch.concat(model(image)).squeeze(1)
    
    
    new_image = place_patch(image, patch, transformation_matrix)
    # pred_attack = torch.concat(model(new_image)).squeeze(1)
    # loss = -torch.dist(prediction, pred_attack, p=2)

    losses = []
    # losses.append(loss.detach().numpy())

    i = 0.
    try:
        while True:
            i += 1
            pred_attack= torch.concat(model(new_image)).squeeze(1)
            loss = -torch.dist(prediction, pred_attack, p=2)
            losses.append(loss.detach().numpy())
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

            # patch.data.clamp_(0., 255.)
            new_image = place_patch(image, patch, transformation_matrix)
            if i % 100 == 0:
                print("step %d, loss %.6f" % (i, loss))
                print("transformation matrix: ", transformation_matrix.detach().numpy())
                #print("step %d, loss %.6f, angle %.2f, scale %.3f, tx %.3f, ty %0.3f" % (i, loss, np.degrees(angle.detach().numpy()), scale.detach().numpy(), tx.detach().numpy(), ty.detach().numpy()))
    except KeyboardInterrupt:
        print("Aborting optimization...")    

    print("Bing!")
    print("Last loss: ", loss.detach().numpy())
    print("Last prediciton: ", pred_attack)

    np.save(path+'losses_test', losses)

    return patch, transformation_matrix #[angle, scale, tx, ty]

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import os

    from util import load_dataset, load_model
    model_path = '../pulp-frontnet/PyTorch/Models/Frontnet160x32.pt'
    model_config = '160x32'
    dataset_path = '../pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(path=model_path, device=device, config=model_config)
    model.eval()
    dataset = load_dataset(path=dataset_path, batch_size=32, shuffle=True, drop_last=True, num_workers=0)

    path = 'eval/targeted_test/'
    os.makedirs(path, exist_ok = True)

    image, pose = dataset.dataset.__getitem__(0)
    image = image.unsqueeze(0)
    
    #attack = TargetedAttack(model, dataset, path=path)
    

    #np.save(path+'ori_patch', attack.x_patch.patch.detach().numpy())

    print("Original pose: ", pose, pose.shape)
    prediction = torch.concat(model(image)).squeeze(1)
    print("Predicted pose: ", prediction)
    print("L2 dist original-predicted: ", torch.dist(pose, prediction, p=2))

    optimized_x_patch, optimized_transformation = targeted_attack(image, torch.tensor([4., *prediction[1:]]), model, path)
    #optimized_x_patch = attack.optimize()
    
    #np.save(path+"opti_patch", optimized_x_patch.patch.detach().numpy())
    #new_image_right = place_patch(image, optimized_x_patch.patch, *optimized_x_patch.patch_parameters_min)
    new_image_right = place_patch(image, optimized_x_patch, optimized_transformation)

    plt.imshow(new_image_right[0][0].detach().numpy(), cmap='gray')
    plt.show()

    # new_image_left = place_patch(image, optimized_x_patch.patch, *optimized_x_patch.patch_parameters_max)

    # plt.imshow(new_image_left[0][0].detach().numpy(), cmap='gray')
    # plt.show()