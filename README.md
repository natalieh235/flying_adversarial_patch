# Adversarial attacks on PULP-Frontnet

## Installation
### Clone the repository
To clone the repository including the code from the pulp-frontnet module, use this command:
```bash
# via ssh
$ git clone --recurse-submodules git@github.com:phanfeld/adversarial_frontnet.git
# or via https
$ git clone --recurse-submodules https://github.com/phanfeld/adversarial_frontnet.git
```

If you have cloned the repository without the `--recurse-submodules` argument and want to pull the pulp-frontnet code, please use the following command inside the repository:
```bash
$ git submodule update --init --recursive
```
### Download the datasets
For downloading the datasets:
```bash
$ cd pulp-frontnet/PyTorch
$ curl https://drive.switch.ch/index.php/s/FMQOLsBlbLmZWxm/download -o pulp-frontnet-data.zip
$ unzip pulp-frontnet-data.zip
$ rm pulp-frontnet-data.zip
```
The datasets should now be located at `pulp-frontnet/PyTorch/Data/`.

### Installing needed Python packages
TODO!

## Compute adversarial patch and position
To generate the adversarial patch with optimal transformation matrices, you can call
```bash
python src/attacks.py --file settings.yaml
```
Please adapt the hyperparameters in `settings.yaml` in the main folder according to your needs.
### Choosing the optimization approach
Please change the optimization approach in the `settings.yaml` file to your desired mode. You can choose between `'fixed'`, `'joint'`, `'split'`, and `'hybrid'`.
```yaml
mode: 'split' # 'fixed', 'joint', 'split', 'hybrid'
```
### Setting target positions
For setting multiple desired target positions $\bar{\mathbf{p}}^h_K$, change the values in `settings.yaml` for the targets like so:
```yaml
  x : [1.0, 0.5]
  y : [-1, 1]
  z : [0, null]
```
Now, the patch will be optimized for two targets: $\bar{\mathbf{p}}^h_1 = (1, -1, 0)^T$, $\bar{\mathbf{p}}^h_2 = (0.5, 1, z)^T$. For the second target, the attack does not set the $z$-value to a desired one but tries to keep it to the originally predicted $z$ for the current image in $\hat{\mathbf{p}}^h$.
### Starting from different initial patch
You can change the initial patch for a training run in the settings file. Either set
```yaml
patch: 
  mode: 'face'
  path: 'src/custom_patches/custom_patch_resized.npy'
```
to, e.g., start from a patch showing a face. Please specify the path to point to a valid numpy array file. The patch should be grayscale and can have any width and height. We prepared multiple patches in the `src/custom_patches` folder.

If the initial patch should be white or starting from random pixel values, adapt the patch mode in the `settings.yaml` like:
```yaml
patch: 
  mode: 'white'
```
or 
```yaml
patch: 
  mode: 'random'
```
### Results
All results will be saved at the specified path in the `settings.yaml`.\
The folder will contain the following files:
```
path
|_settings.yaml # a copy of the settings.yaml
|_patches.npy   # a numpy array containing all patches
|_positions_norm.npy # the optimized positions for the K targets
|_positions_losses.npy # all computed losses for the positions
|_patch_losses.npy # all computed losses for the current patch
|_losses_test.npy # the loss on the entire testset after each iteration
|_losses_test.npy # the loss for the entire trainset after each iteration
|_boxplot_data.npy # an array containing all of the data needed to create the boxplots from the paper
```
## Reproduce the experiments of the paper
To reproduce all of the results from the paper "Flying Adversarial Patches: Manipulating the Behavior of Deep Learning-based Autonomous Multirotors", we prepared several scripts:
### Comparison between the different approaches
To run the full experiment on the different approaches, run:
```bash
python src/exp1.py --file exp1.yaml -j 4 --trials 10 --mode all
```

Please adapt the hyperparameters in the `exp1.yaml` file according to your needs. 

With `-j 4`, 4 worker processes are spawned and approaches are computed in parallel. Depending on your hardware, you can set `-j` to a different value. If `-j` is set to 1, the different approaches will be computed consecutively.

With `--trials 10` you can set the number of paraellel training runs for the same mode to 10 like we did in the paper.

With `--mode all` you can choose all modes ('fixed', 'joint', 'split', 'hybrid'). You can additionally set the mode to one or a combination of all modes with, e.g., `--mode fixed hybrid` to only run the experiment for the 'fixed' and 'hybrid' approach.

The resulting mean test loss for all optimization approaches will be printed in the terminal.\
The results folder will contain a PDF file including the boxplots (among others) similar to Fig. 3 and 4 from the paper.

### Scalability for multiple target positions
To run the experiment on $1\leq K \leq 10$ desired target positions $\bar{\mathbf{p}}^h_K$, run:
```bash
python src/exp2.py --file exp2.yaml -j 4 --trials 10
```

Please adapt `exp2.yaml` according to your needs. Note that the mode needs to be changed in the yaml file! Setting the mode with the `--mode` argument is not possible (currently).

The resulting mean test loss for all $K$ will be printed in the terminal.\
The results folder will contain a PDF file including a plot similar to Fig. 5 from the paper.

### Comparison different starting patches
You can reproduce the experiment analyzing different starting patches with executing:
```bash
python src/exp3.py --file exp3.yaml -j 4 --trials 10 --mode all
```

Please adapt `exp3.yaml` according to your needs.

The resulting mean test loss for all patch modes and optimization approaches will be printed in the terminal.


## Reproduce camera calibration
The camera calibration was performed on the `160x96StrangersTestset` dataset provided by the pulp-frontnet authors. If you followed the steps in [Download the datasets](#download-the-datasets), you can find the dataset here: `pulp-frontnet/PyTorch/Data/160x96StrangersTestset.pickle`.

We saved all the 3D coordinates and the corresponding 2D coordinates of the humans in the images in a csv file. You can find it here: `adversarial_frontnet/camera_calibration/ground_truth_pose.cs`

The 3D coordinates are relative to the camera - the UAV with an attached [AI deck](https://www.bitcraze.io/documentation/tutorials/getting-started-with-aideck/). These values are stored as ground-truth data in the dataset.

The 2D coordinates of the human in the image are manually annotated and therefore prone to errors.

We investigated two ways to ways to calibrate the camera: 
1) calculating a projection matrix with *Direct Linear Transformation* 
2) utilizing OpenCV's `calibrateCamera()` and `projectPoints()` functions

We have calculated the l2 distance between the manually set points stored in the csv file and the calculated pixel coordinates utilizing both methods. The mean l2 distance of the calculated pixel coordinates utilizing OpenCV was smaller. We therefore adapted the OpenCV functions for our code.

You can follow the main method in `adversarial_frontnet/camera_calibration/camera_calibration.py` to calculate the camera intrinsics, rotation and translation matrix and the distortion coefficients needed for projecting pixels. Additionally, you can load these values from the yaml file, provided in the same folder.
