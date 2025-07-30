import os
import time
import torch
import numpy as np
import pandas as pd
import logging
from torch.nn.utils import clip_grad_norm_
from .utils import AverageMeter

logger = logging.getLogger(__name__)

def train_one_epoch(model, dataloader, optimizer, scheduler, criterion, accelerator, epoch, config):
    """
    Train the model for one epoch.
    Args:
        model (torch.nn.Module): The model to train.
        dataloader (DataLoader): DataLoader for the training data.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
        criterion (callable): Loss function.
        accelerator (Accelerator): Accelerator for distributed training.
        epoch (int): Current epoch number.
        config (Config): Configuration object containing training parameters.
    """
    model.train()

    if config.task == 'mlm':

        if accelerator.is_main_process:
            loss_meter = AverageMeter('-')
            lr = optimizer.param_groups[0]['lr']
    
        for batch_idx, (x_pos, x_neg) in enumerate(dataloader):
            start = time.time()
            with accelerator.accumulate(model):
                output = model(x_pos, x_neg, config.mask_probability)
                loss = criterion(output)
                if torch.isnan(loss):
                    nan_processes = [i for i, p_loss in enumerate(accelerator.gather(loss)) if torch.isnan(p_loss)]
                    accelerator.print(f"CRITICAL: Loss became NaN at epoch {epoch}, batch {batch_idx+1} on processes {nan_processes}.")
                    raise RuntimeError("Loss is NaN. Halting training.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            end = time.time()
            
            avg_loss = accelerator.gather(loss).mean().item()
            if accelerator.is_main_process:
                loss_meter.update(avg_loss)
                progress = f"Epoch: {epoch+1}, Batch: [{batch_idx+1:>4}/{len(dataloader):<4}], "
                progress += f"LR: {lr:.8f}, Loss: {loss_meter.val:.6f} (Avg: {loss_meter.avg:.6f}), "
                progress += f"Elapsed: {(end - start)*1000:.2f} ms"
                if batch_idx == len(dataloader) - 1:
                    print(progress, end='\n\r', flush=True)
                else:
                    print(progress, end="\r", flush=True)
        
        if accelerator.is_main_process:
            os.makedirs(config.train_csv_dir, exist_ok=True)
            df = pd.DataFrame({'epoch': [epoch + 1], 'loss': [loss_meter.avg]})
            csv_path = os.path.join(config.train_csv_dir, 'log.csv')
            if epoch != 0 or config.use_pretrained :
                df.to_csv(csv_path, mode='a', header=False, index=False)
            else:
                df.to_csv(csv_path, mode='w', header=True, index=False)

    else:  # Other tasks can be added here
        logger.warning(f"Task '{config.task}' is not implemented. Skipping training for this task.")




