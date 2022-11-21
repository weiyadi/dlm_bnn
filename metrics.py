import numpy as np
from torch import nn
import torch
import math

from torch.nn import functional as F


def logmeanexp(x, dim=None, keepdim=False):
    """Stable computation of log(mean(exp(x))"""

    if dim is None:
        x, dim = x.view(-1), 0
    x_max, _ = torch.max(x, dim, keepdim=True)
    x = x_max + torch.log(torch.mean(torch.exp(x - x_max), dim, keepdim=True))
    return x if keepdim else x.squeeze(dim)


def outputs_to_log_probs_classfication(outputs, weights=None, aggregate='mean'):
    if outputs.dim() == 2:
        # no ensemble
        log_probs = F.log_softmax(outputs, dim=1)
        return log_probs
    else:
        log_probs = F.log_softmax(outputs, dim=2)
        if weights is None:
            if aggregate == 'mean':
                log_probs_aggregate = torch.mean(log_probs, dim=0)
            elif aggregate == 'logmeanexp':
                log_probs_aggregate = logmeanexp(log_probs, dim=0)
            else:
                raise NotImplementedError
            return log_probs_aggregate
        else:
            M = weights.numel()
            L = int(log_probs.size(0) / M)
            B = log_probs.size(1)
            num_classes = log_probs.size(2)
            log_probs = log_probs.reshape((L, M, B, num_classes))
            if aggregate == 'mean':
                log_probs_aggregate = torch.mean(log_probs, dim=0)
                log_probs_final = torch.sum(log_probs_aggregate * weights[:, None, None], dim=0)
            elif aggregate == 'logmeanexp':
                log_probs_aggregate = logmeanexp(log_probs, dim=0)
                log_probs_final = torch.logsumexp(torch.log(weights)[:, None, None] + log_probs_aggregate, dim=0)
            else:
                raise NotImplementedError
            return log_probs_final


def nll_regression(outputs, labels, weights=None, aggregate='mean'):
    if outputs.dim() == 2:
        mean = outputs[:, 0]
        noise = torch.clamp(F.softplus(outputs[:, 1]), min=1e-3, max=1e3)
        nll = 0.5 * ((labels.squeeze() - mean) ** 2 / noise + torch.log(noise) + math.log(2 * math.pi))
        return torch.mean(nll, 0)
    else:
        mean = outputs[:, :, 0]
        noise = torch.clamp(F.softplus(outputs[:, :, 1]), min=1e-3, max=1e3)
        nll = 0.5 * ((labels.squeeze()[None, :] - mean) ** 2 / noise + torch.log(noise) + math.log(2 * math.pi))
        if weights is None:
            if aggregate == 'mean':
                nll_aggregate = torch.mean(nll, dim=0)
            elif aggregate == 'logmeanexp':
                nll_aggregate = -logmeanexp(-nll, dim=0)
            else:
                raise NotImplementedError
            return nll_aggregate
        else:
            M = weights.numel()
            L = int(nll.size(0) / M)
            B = nll.size(1)
            nll = nll.reshape((L, M, B))
            if aggregate == 'mean':
                nll_aggregate = torch.mean(nll, dim=0)
                nll_final = torch.sum(nll_aggregate * weights[:, None], dim=0)
            elif aggregate == 'logmeanexp':
                log_prob_aggregate = logmeanexp(-nll, dim=0)
                nll_final = -torch.logsumexp(torch.log(weights)[:, None] + log_prob_aggregate, dim=0)
            else:
                raise NotImplementedError
            return nll_final


class ELBO(nn.Module):
    def __init__(self, train_size, smooth=None):
        super(ELBO, self).__init__()
        self.train_size = train_size
        self.smooth = smooth

    def forward(self, inputs, target, kl, beta, weights=None, regression=False, *args, **kwargs):
        assert not target.requires_grad
        if not regression:
            log_probs = outputs_to_log_probs_classfication(inputs, weights, aggregate='mean')
            if self.smooth:
                log_probs = torch.log((1 - self.smooth) * torch.exp(log_probs) + self.smooth)
            return F.nll_loss(log_probs, target, reduction='mean') + beta * kl / self.train_size
        else:
            # regression
            nll = nll_regression(inputs, target, weights, aggregate='mean')
            if self.smooth:
                nll = -torch.log((1 - self.smooth) * torch.exp(-nll) + self.smooth)
            return torch.mean(nll, dim=0) + beta * kl / self.train_size


class DLM(nn.Module):
    def __init__(self, train_size, smooth=None):
        super(DLM, self).__init__()
        self.train_size = train_size
        self.smooth = smooth

    def forward(self, inputs, target, kl, beta, weights=None, regression=False, *args, **kwargs):
        assert not target.requires_grad
        if not regression:
            log_probs = outputs_to_log_probs_classfication(inputs, weights, aggregate='logmeanexp')
            if self.smooth:
                log_probs = torch.log((1 - self.smooth) * torch.exp(log_probs) + self.smooth)
            return F.nll_loss(log_probs, target, reduction='mean') + beta * kl / self.train_size
        else:
            nll = nll_regression(inputs, target, weights, 'logmeanexp')
            if self.smooth:
                nll = -torch.log((1 - self.smooth) * torch.exp(-nll) + self.smooth)
            return torch.mean(nll) + beta * kl / self.train_size


def acc(outputs, targets):
    return np.mean(outputs.cpu().numpy().argmax(axis=1) == targets.data.cpu().numpy())


def rmse(outputs, targets):
    if outputs.dim() == 3:
        mean_val = outputs[:, :, 0].detach()
        mean_val = torch.mean(mean_val, dim=0)
    else:
        mean_val = outputs[:, 0].detach()
    return (torch.sqrt(torch.sum((targets.squeeze() - mean_val) ** 2) / targets.size(0))).cpu().numpy()


def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
    return kl


def cross_entropy(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.cross_entropy(output, target)

    return loss, output, {}


def kl_div(model, input, target):
    # standard cross-entropy loss function

    output = model(input)

    loss = F.kl_div(F.log_softmax(output), target, reduction="batchmean")

    return loss, output, {}


def cross_entropy_output(output, target):
    # standard cross-entropy loss function

    loss = F.cross_entropy(output, target)

    return loss, {}


def nll(model, input, target):
    output = model(input)
    nll = nll_regression(output, target)
    return torch.mean(nll), output, {}


def eval_batch(outputs, labels, regression=False):
    if isinstance(outputs, list):
        outputs = torch.stack(outputs)
    if not regression:
        log_prob = outputs_to_log_probs_classfication(outputs, aggregate='logmeanexp')
        loss = F.nll_loss(log_prob, labels, reduction='mean').cpu().data.numpy()
        acc_val = acc(log_prob, labels)
    else:
        nll = nll_regression(outputs, labels, aggregate='logmeanexp')
        loss = torch.mean(nll, dim=0)
        acc_val = rmse(outputs, labels)
    return loss, acc_val
