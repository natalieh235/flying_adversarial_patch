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

