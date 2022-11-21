import models
import argparse
import torch
from helpers import get_kwargs, eval_epoch_vi_criterion, eval_epoch_vi
import data
from posteriors import VIFFGModel
import numpy as np
from metrics import ELBO, DLM
import matplotlib.pyplot as plt
import matplotlib
import yaml
import math

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description='vi model training')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
    parser.add_argument('--net_type', default='AlexNet', type=str, help='neural network name')
    parser.add_argument('--prior_var', type=float, default=0.05, help='The prior var')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for data')
    parser.add_argument('--beta', type=float, default=0.1, help='the regularization parameter')
    parser.add_argument('--model_path1', type=str, help='Path for the first model')
    parser.add_argument('--model_path2', type=str, help='Path for the second model')

    # yaml configuration
    parser.add_argument('--config', type=str, default='', help='configurations')

    args = parser.parse_args()

    print('Using model %s' % args.net_type)
    model_cfg = getattr(models, args.net_type)

    trainset, valset, inputs, outputs = data.getTransformedDataset(args.dataset, model_cfg)
    num_classes = outputs

    dataset = args.dataset
    net_type = args.net_type

    torch.backends.cudnn.benchmark = True

    # load general configuration
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)
    valid_size = cfg['valid_size']
    batch_size = args.batch_size
    num_workers = cfg['num_workers']
    epochs = cfg['n_epochs']
    L_train = cfg['L_train']
    L_test = cfg['L_test']
    save_interval = cfg['save_interval']
    train_loader, _, test_loader = data.getDataloader(trainset, valset, valid_size, batch_size, num_workers)

    model_args = list()
    regression = False
    model_kwargs = get_kwargs(args.dataset, model_cfg, regression, inputs, outputs)
    priors = cfg['priors']
    priors['prior_sigma'] = math.sqrt(args.prior_var)

    beta = args.beta

    # Load the models
    state_dict1 = torch.load(args.model_path1)
    mu1 = state_dict1['mu'].cpu().numpy()
    rho1 = state_dict1['rho'].cpu().numpy()

    state_dict2 = torch.load(args.model_path2)
    mu2 = state_dict2['mu'].cpu().numpy()
    rho2 = state_dict2['rho'].cpu().numpy()

    vi_model = VIFFGModel(model_cfg.base, priors, 'naive', cfg, *model_args, **model_kwargs).to(device)
    [p.requires_grad_(False) for p in vi_model.parameters()]

    margin = 0.2
    alphas = np.arange(0.0 - margin, 1.0 + margin, 0.1)

    criterion_elbo = ELBO(len(train_loader.dataset))
    elbo = np.zeros(alphas.size)
    criterion_dlm = DLM(len(train_loader.dataset))
    dlm = np.zeros(alphas.size)
    test = np.zeros(alphas.size)
    kl = np.zeros(alphas.size)

    for i, alpha in enumerate(alphas):
        p_mu = (1 - alpha) * mu1 + alpha * mu2
        p_rho = (1 - alpha) * rho1 + alpha * rho2
        vi_model.mu.copy_(torch.from_numpy(p_mu))
        vi_model.rho.copy_(torch.from_numpy(p_rho))
        elbo_result = eval_epoch_vi_criterion(vi_model, criterion_elbo, train_loader, L=cfg['L_train'], beta=beta,
                                              regression=regression)
        elbo[i] = elbo_result[0]
        dlm[i] = eval_epoch_vi_criterion(vi_model, criterion_dlm, train_loader, L=cfg['L_train'], beta=beta,
                                         regression=regression)[0]
        test[i] = eval_epoch_vi(vi_model, test_loader, L=cfg['L_test'], regression=regression)[0]
        kl[i] = elbo_result[1] / len(train_loader.dataset)

    plt.figure()
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    plt.plot(alphas, elbo, '-o', color=colors[0], label='elbo')
    plt.plot(alphas, dlm, '-s', color=colors[1], label='dlm')
    plt.plot(alphas, test, '-D', color=colors[2], label='test')
    plt.plot(alphas, elbo - beta * kl, '-.v', color=colors[0], label='elbo(no kl)')
    plt.plot(alphas, dlm - beta * kl, '-.^', color=colors[1], label='dlm(no kl)')

    plt.margins(0.0)
    plt.xlabel('alpha')
    plt.legend(loc='best')
    plt.savefig('loss_surface-{}-{}-prior{}.pdf'.format(args.dataset, args.net_type, args.prior_var))
