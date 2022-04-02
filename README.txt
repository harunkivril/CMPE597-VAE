The scripts uses relative paths therefore please run them from the directory that contains the codes.

INSTALLATION:
=============
Required non default packages for python:
pytorch
torchvision
numpy
matplotlib
tqdm

Also it is possible to create a conda environment with the environment.yml usign the following commands:

conda env create -f environment.yml
conda activate assignment

Using the conda environment also makes you able to use the same version with the author. 
If you are don't have conda you may want to install the compact version from the following link:
https://docs.conda.io/en/latest/miniconda.html

If you have gpu please consider changing "- pytorch=1.8.0" to "- pytorch-gpu=1.8.0" since it makes everything faster.

USAGE:
======
When "main.py" is run it automaticaly downloads data to ./data folder and trains a model with the parameters defined at the begining of of the script
It saves the state dict of best model to path defined as save_folder.
Save folder also will contain loss plot, reconstructed images, test results and the config file
`python main.py`
or
`python3 main.py`
commands is enough to start it

When "generator.py" is called with similar python command, it checks for "./best_model.pkl"  loads it generates 100 images and saves to "./generated_images.png"

"model.py" contains the model class.
