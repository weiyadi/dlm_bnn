import torch


def set_weights(model, vector, device=None):
    offset = 0
    for name, param in model.named_parameters():
        param.copy_(vector[offset:(offset + param.numel())].view(param.size()).to(device))
        offset += param.numel()


def kl(mu, sigma, prior_mu, prior_sigma, log_prior_sigma=None):
    if log_prior_sigma is None:
        log_prior_sigma = torch.log(prior_sigma)
    kl = 0.5 * (2 * (log_prior_sigma - torch.log(sigma)) - 1 + (sigma / prior_sigma).pow(2) + (
            (mu - prior_mu) / prior_sigma).pow(2)).sum()
    return kl


def collapsed_kl_mean(mu, sigma, gamma, alpha_reg, log_sigma=None, const=None):
    if log_sigma is None:
        log_sigma = torch.log(sigma)
    if const is None:
        const = 0.5 * torch.log(gamma) - 0.5
    result = 0.5 / gamma * (sigma ** 2 + alpha_reg * mu ** 2) - log_sigma - 0.5 * torch.log(alpha_reg) + const
    return torch.sum(result)


def collapsed_kl_mv(mu, sigma, alpha, beta, delta, log_sigma=None, const=None):
    if log_sigma is None:
        log_sigma = torch.log(sigma)
    if const is None:
        const = -torch.lgamma(alpha + 0.5) + torch.lgamma(alpha) - alpha * torch.log(beta) - torch.log(delta)
    result = (alpha + 0.5) * torch.log(beta + 0.5 * delta * mu ** 2 + 0.5 * sigma ** 2) - log_sigma + const
    return torch.sum(result)


def eb_kl(mu, sigma, alpha, beta):
    if mu.dim() == 2:
        # uq part
        N = mu.size(1)
        s_star = (torch.sum(mu ** 2 + sigma ** 2, 1) + 2 * beta) / (N + 2 * alpha + 2)
        s_star = s_star.reshape(-1, 1)
    else:
        # vi part
        N = mu.numel()
        s_star = (torch.sum(mu ** 2 + sigma ** 2) + 2 * beta) / (N + 2 * alpha + 2)
    log_prior_sigma = 0.5 * torch.log(s_star)
    kl_val = kl(mu, sigma, 0, torch.sqrt(s_star), log_prior_sigma=log_prior_sigma)
    return torch.sum(kl_val)
