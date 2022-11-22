# Direct Loss Minimization for Bayesian Neural Networks
This repository includes the basic Python code for the following paper:

Y. Wei and R. Khardon. On the Performance of Direct Loss Minimization for Bayesian Neural Networks
https://arxiv.org/abs/2211.08393

## Directory structure

This repository has the following directory structure
 * *README*: This file.
 * *data*: prepares data.
 * *example*: contains two example model file and one configuration file for running the demonstration below.
 * *models*: contains the implementation of the neural networks.
 * *posteriors*: contains the implementation of variational inference.
 * *helpers.py*: contains the helper functions to train and evaluate of the models.
 * *metrics.py*: contains the implementation of ELBO and DLM objectives.
 * *run_vi.py*: the script to perform variational inference. The choice of ELBO or DLM is decided by the argument.
 * *run_vi_bo.py*: the script to perform variational inference with bounded optimization.
 * *loss_surface.py*: the script to produce the loss surface between two models.
 * *requirements.txt*: requirements of python packages.

## Run variational inference with ELBO or DLM
It is better to run everything on a GPU for faster speed. On a V100 GPU, it would normally take 1min for each epoch.
It is not recommended to run the script with large image datasets and complicated neural networks on CPU because it would take forever.
We run `run_vi.py` to train a model with different loss functions.  
`--dataset` sets the dataset, CIFAR10, CIFAR100, SVHN or STL10; 
`--net_type` sets the neural network structure, AlexNet or PreResNet20;
`--metric` sets the loss function, ELBO or DLM;
`--seed` sets the random seed;
`--beta` sets the regularization parameter, default 0.1;
`--reg` sets the type of the regularization, naive (use the original kl divergence), 
mean (collapsed variational inference that only learns prior mean), 
mv (collapsed variational inference that learns both prior mean and variance),
or eb (empirical bayes);
`--batch_size` sets the size of batch, default 512;
`--prior_var` sets the prior variance, default 0.05;
`--save_path` sets the directory to save the models, default 'checkpoints'';
`--init_path` is the path of the model to initialize;
`--smooth` sets the smooth parameter for the loss function, default 0, meaning no smoothing;
`--config` is the path for configuration file that contains learning rate, training epochs, sample_sizes, etc, 
we provide `example/example.yaml` as an example.
Below is an example to run this script without initialization:
```console
python run_vi.py --dataset CIFAR10 --net_type AlexNet --metric ELBO --seed 1 --beta 0.1 --reg naive --batch_size 512 \
--prior_var 0.05 --save_path checkpoints --config example/example.yaml
```
and an example to run with initialization:
```console
python run_vi.py --dataset CIFAR10 --net_type AlexNet --metric ELBO --seed 1 --beta 0.1 --reg naive --batch_size 512 \
--prior_var 0.05 --save_path checkpoints --init_path example/elbo.pt --config example/example.yaml
```
Users can freely change arguments including the dataset, neural network and the configuration file. 
The above scripts will produce several `.pt` files that save the model state dictionary at the certain epochs, 
depending on the choice of `save_interval` in the config file, 
and one `.npz` file that saves the statistics along optimization, including train_obj (elbo if metric selects ELBO, otherwise dlm), 
train_acc, train_kl, test_loss, test_acc, train_loss, other_obj (dlm if metric selects ELBO, otherwise elbo). 
The `.pt` files can be used to produce loss surfaces. The `.npz` file can be used to produce learning curves.

## Run variational inference using bounded optimization
We run `run_vi_bo.py` to train a model using bounded optimization with either ELBO or DLM. The arguments are similar to 
those of `run_vi.py` except no `--reg`, `--prior_var`, `--beta`, but add
`--bm` to set the l2-norm bound for posterior mean and `--bv` to set the infinity-norm bound for posterior variance.
For example,
```console
python run_vi_bo.py --dataset CIFAR10 --net_type AlexNet --metric ELBO --seed 1 --batch_size 512 --bm 50 --bv 120 \
--save_path checkpoints --init_path example/elbo.pt --config example/example.yaml
```

## Loss surface
`loss_surface.py` is provided to produce the loss surface between two models. 
`--dataset`, `--net_type`, `--batch_size`, `--prior_var` and `--beta` all have the same meanings as before.
`--model_path1` and `--model_path2` specify the paths of two models.
```console
python loss_surface.py --dataset CIFAR10 --net_type AlexNet --batch_size 512 --prior_var 0.05 --beta 0.1 \
--model_path1 example/elbo.pt --model_path2 example/dlm.pt --config example/example.yaml
```
The figure will be saved as `loss_surface-CIFAR10-AlexNet-prior0.05.pdf`. It will take ~20 mins to finish on a V100 GPU.