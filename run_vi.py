import argparse
import os

import torch
import math

from posteriors import VIFFGModel
import data
import numpy as np
import models
import metrics
from helpers import train_epoch_vi, get_kwargs, eval_epoch_vi, eval_epoch_vi_criterion
import yaml

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    parser = argparse.ArgumentParser(description='vi model training')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
    parser.add_argument('--net_type', default='AlexNet', type=str, help='neural network name')
    parser.add_argument('--metric', default='ELBO', type=str, help='metric', choices=['ELBO', 'DLM'])
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

    parser.add_argument('--beta', type=float, default=0.1, help='regularization parameter')
    parser.add_argument('--reg', type=str, default='naive', help='regularization type')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size for data')
    parser.add_argument('--prior_var', type=float, default=0.05, help='The prior variance')
    parser.add_argument('--save_path', type=str, default='checkpoints', help='Directory to save result')
    parser.add_argument('--init_path', type=str, default='', help='the path of model to initialize')
    parser.add_argument('--smooth', type=float, default=0., help='smooth factor of log loss')

    # yaml configuration
    parser.add_argument('--config', type=str, default='', help='configurations')

    args = parser.parse_args()

    print('Using model %s' % args.net_type)
    model_cfg = getattr(models, args.net_type)

    dataset = args.dataset
    net_type = args.net_type
    seed = args.seed
    torch.manual_seed(args.seed)

    torch.backends.cudnn.benchmark = True

    # load general configuration
    with open(args.config, 'r') as file:
        cfg = yaml.safe_load(file)
    print("ALL CONFIGURATION:", cfg)

    valid_size = cfg['valid_size']
    batch_size = args.batch_size
    num_workers = cfg['num_workers']
    epochs = cfg['n_epochs']
    L_train = cfg['L_train']
    L_test = cfg['L_test']
    save_interval = cfg['save_interval']
    beta = args.beta
    regression = cfg['regression']

    trainset, valset, inputs, outputs = data.getTransformedDataset(dataset, model_cfg)
    num_classes = outputs
    # prepare ensembles
    train_loader, _, test_loader = data.getDataloader(trainset, valset, valid_size, batch_size, num_workers)

    print('Preparing model')
    model_args = list()
    model_kwargs = get_kwargs(args.dataset, model_cfg, regression, inputs, outputs)
    print('Model args:', model_kwargs)
    priors = cfg['priors']
    priors['prior_sigma'] = math.sqrt(args.prior_var)

    base_model = model_cfg.base(*model_args, **model_kwargs).to(device)
    pos_mean, pos_rho = None, None

    init_beta = 0.1
    if args.init_path != '':
        model_state_dict = torch.load(args.init_path)
        pos_mean, pos_rho = model_state_dict['mu'], model_state_dict['rho']

    vi_model = VIFFGModel(model_cfg.base, priors, args.reg, cfg, pos_mean=pos_mean, pos_rho=pos_rho, *model_args,
                          **model_kwargs).to(device)
    os.makedirs(args.save_path, exist_ok=True)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vi_model.parameters()), lr=cfg['lr_start'])
    scheduler = None

    if args.metric == 'DLM':
        criterion = metrics.DLM(len(train_loader.dataset), args.smooth)
        other_criterion = metrics.ELBO(len(train_loader.dataset), args.smooth)
    else:
        criterion = metrics.ELBO(len(train_loader.dataset), args.smooth)
        other_criterion = metrics.DLM(len(train_loader.dataset), args.smooth)

    opt_traj = []
    for epoch in range(epochs):
        train_obj, _, train_kl = train_epoch_vi(vi_model, optimizer, criterion, train_loader,
                                                L=L_train, beta=beta, regression=regression, scheduler=scheduler)
        other_obj, _ = eval_epoch_vi_criterion(vi_model, other_criterion, train_loader, L=L_train, beta=beta,
                                               regression=regression)
        train_loss, train_acc = eval_epoch_vi(vi_model, train_loader, L=L_test, regression=regression)
        test_loss, test_acc = eval_epoch_vi(vi_model, test_loader, L=L_test, regression=regression)

        opt_traj.append([train_obj, train_acc, train_kl, test_loss, test_acc, train_loss, other_obj])

        print(
            'Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
                epoch, train_obj, train_acc, test_loss, test_acc, train_kl))

        if epoch % save_interval == 0 or epoch == epochs - 1:
            torch.save(vi_model.state_dict(), args.save_path + f'/epoch{epoch}.pt')
    np.savez(args.save_path + f'/result', opt_traj=opt_traj)
