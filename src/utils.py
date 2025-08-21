import torch
import torch.nn as nn
import torch.optim as optim
import bitsandbytes as bnb
from transformers.optimization import get_scheduler
import numpy as np
import random
import os

def build_optimizer(model, config):
    """
    Build an optimizer for the model.
    Args:
        model (torch.nn.Module): The model to optimize.
        config (Config): Configuration object containing optimizer parameters.
    Returns:
        torch.optim.Optimizer: The optimizer instance.
    """
    param = filter(lambda p:p.requires_grad, model.parameters())
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(param, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'adam':
        optimizer = optim.Adam(param, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'sgd':
        optimizer = optim.SGD(param, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)
    elif config.optimizer == 'adamw8bit':
        optimizer = bnb.optim.AdamW8bit(param, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'lion':
        optimizer = bnb.optim.Lion(param, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))
    elif config.optimizer == 'lion8bit':
        optimizer = bnb.optim.Lion8bit(param, lr=config.learning_rate, weight_decay=config.weight_decay, betas=(config.adam_beta1, config.adam_beta2))     
    else:
        optimizer = optim.SGD(param, lr=config.learning_rate, weight_decay=config.weight_decay, momentum=config.momentum)

    return optimizer

def build_scheduler(optimizer, config, steps_per_epoch=1):
    """
    Build a learning rate scheduler for the optimizer.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to schedule.
        config (Config): Configuration object containing scheduler parameters.
        steps_per_epoch (int): Number of steps per epoch for the scheduler from len(dataloader).
    Returns:
        torch.optim.lr_scheduler: The learning rate scheduler instance.
    """
    scheduler_specific_kwargs = {}
    if config.scheduler == 'cosine_with_restarts':
        scheduler_specific_kwargs = {
            'num_cycles': config.num_cycles,
        }
    elif config.scheduler == 'cosine_with_min_lr':
        scheduler_specific_kwargs = {
            'min_lr': config.min_lr_rate
        }
    elif config.scheduler == 'polynomial':
        scheduler_specific_kwargs = {
            'power': config.power
        }
    all_steps = (config.max_epochs * steps_per_epoch) // config.accumulate + 1
    scheduler = get_scheduler(
        config.scheduler,
        optimizer,
        num_warmup_steps=int(all_steps * config.warmup_proportion),
        num_training_steps=all_steps,
        scheduler_specific_kwargs= scheduler_specific_kwargs
    )

    return scheduler

class AverageMeter(object):
    """ Computes and stores the average and current value.
    """
    def __init__(self, unit='-'):
        self.reset()
        self.unit = unit

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val    = val
        self.sum   += val * n
        self.count += n
        self.avg    = self.sum / self.count if self.count != 0 else 0

class AverageSeq(object):
    """ Computes and stores the average of sequences at each position. """
    def __init__(self, length):
        self.memory = np.zeros(length)
        self.count = np.zeros(length)

    def update(self, seq, startid):
        for i in range(seq):
            self.memory[startid + i] = seq[i]
            self.count[startid + i] += 1

    def avg(self):
        """Returns the average values for each position, handling zero counts."""
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_values = np.true_divide(self.memory, self.count)
            avg_values[self.count == 0] = 0  # Set average to 0 where count is 0
        return avg_values

def set_all_seeds(config):
    # Set seeds
    os.environ['PYTHONHASHSEED'] = str(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.seed)
        torch.backends.cudnn.benchmark     = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled       = True

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.xavier_uniform_(m.weight)
    # elif isinstance(m, YourCustomLayer):
    #     pass
