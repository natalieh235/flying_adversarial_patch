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
### Setting up a Python Virtual Environment
Please make sure you have Python >= 3.7.10 installed.
```bash
$ python -m venv /path/to/env
$ source path/to/env/bin/activate
$ python -m pip install -r path/to/repo/adversarial-frontnet/requirements.txt
```

To add the virtual environment as a kernel for Jupyter Notebook
```bash
$ python -m pip install ipykernel
$ python -m ipykernel install --user --name=kernel_name
```
<!-- ### Anaconda Virtual Environment -->

<!-- ### GAP SDK 3.9.1
* Download release from https://github.com/GreenWaves-Technologies/gap_sdk/releases/tag/release-v3.9.1
*  -->

## Usage
To perform an attack, you'll currently need to work with `adversarial_frontnet/attacks.py`.\
Please adjust the code in the main routine for your current use-case.
### Example for the creation of a single patch for a single image
(1) Choose an image from the dataset, here we chose image 70:
```python
image = dataset.dataset.data[70].unsqueeze(0).to(device)
```
(2) Calculate a patch, that shifts the prediction of the x position to 3., on 10,000 epochs and a learning rate of 1e-4: 
```python
patch = single_image_attack(model, image, 3., device, 10000, lr=1e-4)
```
(3) Specify a path for saving the raw data of the patch with numpy (without the .npy file ending)
```python
np.save('path/to/file', patch)
```
### Example for creating multiple patches for multiple images
(1) You don't need to specify an image to attack. Simply hand over the whole dataset to the function:
```python
patches = multi_image_attack(model=model, train_data=dataset, target_value=3., device=device)
```
(2) Specify a path for saving the raw data of the patches with numpy (without the .npy file ending)
```python
np.save('path/to/file', patchs)
```
Important! This method currently does not produce a reasonable, universal attack!

### Example for placing a single patch at an arbitrary position
Please follow the main method in `adversarial_frontnet/patch_placement.py` to place a random patch in a single image at an arbitrary position.

Prerequisites: Load or calculate the camera configuration (see [Reproduce camera calibration](#reproduce-camera-calibration))

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

## Hardware
### Generate C code and flashable image of quantized Frontnet
For creating a flashable image, we first need a `.onnx` file of the quantized networks. We use [nemo](https://github.com/pulp-platform/nemo) to receive the `.onnx` file.\
Nemo only works on older versions of PyTorch. We therefore create a new Python Virtual Environment for this process.
```bash
deactivate  # deactive your current virtual environment
python3 -m venv /path/to/nemo-env
source /path/to/nemo-env/bin/activate
python -m pip install torch==1.4.0 torchvision==0.5.0 pytorch-nemo==0.0.7 pandas==1.2.4 torchsummary==1.5.1 matplotlib==3.4.1
```

After installing nemo, please change directories back to the root directory of this repository.

You can now call the script, that the Frontnet authors provided for generating the `.onnx` file:
```bash
python pulp-frontnet/PyTorch/Scripts/QExport.py '160x32' --load-model 'pulp-frontnet/PyTorch/Models/Frontnet160x32.Q.pt' --load-trainset 'pulp-frontnet/PyTorch/Data/160x96OthersTrainsetAug.pickle' --regime pulp-frontnet/PyTorch/Scripts/regime.json
```

This will create a new folder `Results/160x32/Export` in which the `Frontnet.onnx` is saved.

For generating the flashable image, we'll utilize the Bitcraze AI deck docker image, since building depends partially on the GAP SDK.\
Run the docker image and mount this repository
```bash
docker run -it -v /path/to/adversarial_frontnet/:/home/adversarial_frontnet bitcraze/aideck
source /gap_sdk/configs/ai_deck.sh
cd /home/
```

We use DORY to generate a flashable image from the `.onnx` file.
```bash
# clone the repository
git clone https://github.com/pulp-platform/dory
cd dory
# we tested dory on this commit
git checkout 06b1b91fe1aa77f87b3baae97ee8dcb03eef1785
# get submodules
git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn
git submodule update --remote --init dory/Hardware_targets/GAP8/Backend_Kernels/pulp-nn-mixed
# install DORY as pip pickage into your current python environment
python -m pip install -e .
```

Now generate the image with the provided script:
```bash
python network_generate.py NEMO GAP8.GAP8_gvsoc /home/adversarial_frontnet/misc/dory_config.json --app_dir /home/adversarial_frontnet/hardware/frontnet_code/
```

TODO: Add section about adapting the source code before building!

Lastly, to generate the C code and flashable image:
```bash
cd /home/adversarial_frontnet/hardware/frontnet_code/
make clean all run CORE=8 platform=gvsoc
```


### ROS2 package

## ROS2 Package

Use a symlink to your ROS2 workspace

```
ln -s <path-to>/adverserial_frontnet/hardware/frontnet_ros <path-to>/ros2_workspace/src
```