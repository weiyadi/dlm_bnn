import torch
from torch.nn import functional as F

import metrics
import numpy as np
import inspect

from metrics import outputs_to_log_probs_classfication, nll_regression, rmse, logmeanexp


def inv_softmax(x, eps=1e-10):
    return torch.log(x / (1.0 - x + eps))


def train_epoch_vi(vi_model, optimizer, criterion, trainloader, device=None, L=1, beta=0.1, regression=False,
                   scheduler=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vi_model(L, inputs)

        kl = vi_model.compute_reg(inputs)
        kl_list.append(kl.item())

        if not regression:
            if hasattr(vi_model, 'log_weights'):
                loss = criterion(outputs, labels, kl, beta, weights=F.softmax(vi_model.log_weights, dim=0))
            else:
                loss = criterion(outputs, labels, kl, beta)
            loss.backward()

            log_outputs = F.log_softmax(outputs, dim=2)
            log_output = logmeanexp(log_outputs, dim=0)
            accs.append(metrics.acc(log_output.data, labels))
        else:
            loss = criterion(outputs, labels, kl, beta, regression=regression)
            loss.backward()

            accs.append(metrics.rmse(outputs, labels))
        # print(loss.item())
        training_loss += loss.cpu().data.numpy()
        nonan = True
        for name, param in vi_model.named_parameters():
            if param.requires_grad and torch.any(torch.isnan(param.grad)):
                nonan = False
                break
        if nonan:
            optimizer.step()
            if scheduler:
                scheduler.step()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list)


def train_epoch_vi_bo(vi_model, optimizer, criterion, trainloader, bm=None, bv=None, device=None, L=1, regression=False,
                      scheduler=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = vi_model(L, inputs)

        kl = vi_model.compute_reg(inputs)
        kl_list.append(kl.item())

        if not regression:
            if hasattr(vi_model, 'log_weights'):
                loss = criterion(outputs, labels, kl, beta=0., weights=F.softmax(vi_model.log_weights, dim=0))
            else:
                loss = criterion(outputs, labels, kl, beta=0.)
            loss.backward()

            log_outputs = F.log_softmax(outputs, dim=2)
            log_output = logmeanexp(log_outputs, dim=0)
            accs.append(metrics.acc(log_output.data, labels))
        else:
            loss = criterion(outputs, labels, kl, beta=0., regression=regression)
            loss.backward()

            accs.append(metrics.rmse(outputs, labels))
        # print(loss.item())
        training_loss += loss.cpu().data.numpy()
        nonan = True
        for name, param in vi_model.named_parameters():
            if param.requires_grad and torch.any(torch.isnan(param.grad)):
                nonan = False
                break
        if nonan:
            optimizer.step()

            if torch.sqrt(torch.sum(vi_model.mu ** 2)) > bm:
                vi_model.mu.data = vi_model.mu / torch.sqrt(torch.sum(vi_model.mu ** 2)) * bm
            vi_model.rho.data = torch.clamp(vi_model.rho, max=inv_softmax(torch.tensor(bv)))
            if scheduler:
                scheduler.step()
    return training_loss / len(trainloader), np.mean(accs), np.mean(kl_list)


def eval_epoch_vi_criterion(vi_model, criterion, trainloader, device=None, L=1, beta=0.1, regression=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    training_loss = 0.0
    with torch.no_grad():
        kl = vi_model.compute_reg()
        for i, (inputs, labels) in enumerate(trainloader, 1):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = vi_model(L, inputs)
            if not regression:
                if hasattr(vi_model, 'log_weights'):
                    loss = criterion(outputs, labels, kl, beta, weights=F.softmax(vi_model.log_weights, dim=0))
                else:
                    loss = criterion(outputs, labels, kl, beta)
            else:
                loss = criterion(outputs, labels, kl, beta, regression=regression)
            # print(loss.item())
            training_loss += loss.cpu().data.numpy()
    return training_loss / len(trainloader), kl.item()


def eval_epoch_vi(vi_model, testloader, device=None, L=1, regression=False):
    """Calculate accuracy and NLL Loss"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    vi_model.train()
    test_loss = 0.0
    accs = []
    with torch.no_grad():
        for i, (input, labels) in enumerate(testloader):
            input, labels = input.to(device), labels.to(device)
            outputs = vi_model(L, input)
            if hasattr(vi_model, 'log_weights'):
                weights = F.softmax(vi_model.log_weights)
            else:
                weights = None
            if not regression:
                log_output = outputs_to_log_probs_classfication(outputs, weights, aggregate='logmeanexp')
                test_loss += (F.nll_loss(log_output, labels, reduction='mean')).cpu().data.numpy()
                accs.append(metrics.acc(log_output.data, labels))
            else:
                nll = nll_regression(outputs, labels, weights, aggregate='logmeanexp')
                test_loss += torch.mean(nll)
                accs.append(metrics.rmse(outputs, labels))
    return test_loss / len(testloader), np.mean(accs)


def eval_epoch_vi_ensemble(vi_models, testloader, device=None, L=1, regression=False):
    """Calculate ensemble accuracy and NLL Loss"""
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_loss = 0.0
    accs = []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            all_log_probs = []
            rmses = []
            for vi_model in vi_models:
                outputs = vi_model(L, inputs)
                if hasattr(vi_model, 'log_weights'):
                    weights = F.softmax(vi_model.log_weights)
                else:
                    weights = None
                if not regression:
                    log_prob = outputs_to_log_probs_classfication(outputs, weights, aggregate='logmeanexp')
                    all_log_probs.append(log_prob)
                else:
                    nll = nll_regression(outputs, labels, weights, aggregate='logmeanexp')
                    all_log_probs.append(-nll)
                    rmses.append(rmse(outputs, labels))
            log_prob_final = logmeanexp(torch.stack(all_log_probs), dim=0)
            if not regression:
                test_loss += (F.nll_loss(log_prob_final, labels, reduction='mean')).cpu().data.numpy()
                accs.append(metrics.acc(log_prob_final.data, labels))
            else:
                test_loss += -torch.mean(log_prob_final)
                accs.append(np.mean(rmses))

    return test_loss / len(testloader), np.mean(accs)


def get_vi_predictions(model, loader, L=1, regression=False, device=None, prob=True):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_values = []
    if not regression:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)
                if L is None:
                    outputs = model(inputs)
                else:
                    outputs = model(L, inputs)
                log_probs = outputs_to_log_probs_classfication(outputs, aggregate='logmeanexp')
                nll = F.nll_loss(log_probs, labels, reduction='none')
                if prob:
                    all_values.append(torch.exp(-nll).cpu().numpy())
                else:
                    all_values.append(nll.cpu().numpy())
            all_values = np.concatenate(all_values, axis=0)
        return all_values
    else:
        raise NotImplementedError


def get_predictions_labels(models, loader, L=1, regression=False, device=None, swag=False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    all_probs = []
    all_log_probs = []
    all_labels = []
    if not regression:
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = []
                if isinstance(models, list):
                    for model in models:
                        if L is not None:
                            output = model(L, inputs)
                        else:
                            output = model(inputs)
                        outputs.append(output)
                    outputs = torch.stack(outputs)
                elif swag:
                    for _ in range(L):
                        models.sample(1.0)
                        output = models(inputs)
                        outputs.append(output)
                    outputs = torch.stack(outputs)
                else:
                    if L is not None:
                        outputs = models(L, inputs)
                    else:
                        outputs = models(inputs)
                log_probs = outputs_to_log_probs_classfication(outputs, aggregate='logmeanexp')
                all_log_probs.append(log_probs.cpu().numpy())
                probs = torch.exp(log_probs)
                all_probs.append(probs.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        all_probs = np.concatenate(all_probs, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_log_probs = np.concatenate(all_log_probs, axis=0)
        return all_probs, all_labels, all_log_probs
    else:
        raise NotImplementedError


def get_kwargs(dataset, model_cfg, regression, *args):
    kwargs = model_cfg.kwargs
    if 'in_channels' in inspect.getfullargspec(model_cfg.base.__init__).args:
        if dataset == 'MNIST':
            kwargs['in_channels'] = 1
        else:
            kwargs['in_channels'] = 3
    if not regression:
        kwargs['num_classes'] = args[1]
        if 'mlp' in str(model_cfg):
            kwargs['in_dim'] = args[0]
    else:
        kwargs['in_dim'] = args[0]
    return kwargs
