# Core dependencies
# Install PyTorch & torchvision first!!! - otherwise it will be installed as a transitive dependency and probably without GPU support
pip install matplotlib tabulate wilds netcal cw2 transformers wandb laplace-torch
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install git+https://github.com/treforevans/uci_datasets.git

# HMC samples
pip install jax==0.4.1 jaxlib==0.4.1+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install tensorflow tensorflow_datasets
pip install dm-haiku
conda install -c nvidia cuda-nvcc
