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

    if config.task == 'mlm':

        if accelerator.is_main_process:
            loss_meter = AverageMeter('-')
            lr = optimizer.param_groups[0]['lr']

        for batch_idx, x_seq in enumerate(dataloader):
            start = time.time()
            with accelerator.accumulate(model):
                output = model(x_seq, config.mask_probability)
                loss = criterion(output)
                if torch.isnan(loss):
                    accelerator.print(f"CRITICAL: Loss became NaN at epoch {epoch}, batch {batch_idx+1}.")
                    raise RuntimeError("Loss is NaN. Halting training.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            end = time.time()
            
            if accelerator.is_main_process:
                loss_meter.update(loss.item())
                progress = f"Epoch: {epoch+1}, Batch: [{batch_idx+1:>4}/{len(dataloader):<4}], "
                progress += f"LR: {lr:.8f}, Loss: {loss_meter.val:.6f} (Avg: {loss_meter.avg:.6f}), "
                progress += f"Elapsed: {(end - start)*1000:.2f} ms"
                if batch_idx == len(dataloader) - 1:
                    print(progress, end='\n\r', flush=True)
                else:
                    print(progress, end="\r", flush=True)

        if accelerator.is_main_process:
            log_data = {
                'epoch': epoch + 1,
                'loss': loss_meter.avg
            }

    elif config.task == 'traj':
        if accelerator.is_main_process:
            loss_position_meter = AverageMeter('-')
            loss_velocity_meter = AverageMeter('-')
            loss_rotation_meter = AverageMeter('-')
            lr = optimizer.param_groups[0]['lr']
        for batch_idx, (x_seq, traj_seq) in enumerate(dataloader):
            start = time.time()
            traj_seq_w = traj_seq[:, :, 6:]
            with accelerator.accumulate(model):
                output = model(x_seq, traj_seq_w)
                loss_pos, loss_vel, loss_rot = criterion((traj_seq, output))
                loss = config.a_pos * loss_pos + config.a_vel * loss_vel + config.a_rot * loss_rot
                if torch.isnan(loss):
                    accelerator.print(f"CRITICAL: Loss became NaN at epoch {epoch}, batch {batch_idx+1}.")
                    raise RuntimeError("Loss is NaN. Halting training.")
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            end = time.time()
            
            if accelerator.is_main_process:
                loss_position_meter.update(loss_pos.item())
                loss_velocity_meter.update(loss_vel.item())
                loss_rotation_meter.update(loss_rot.item())
                progress = f"Epoch: {epoch+1}, Batch: [{batch_idx+1:>4}/{len(dataloader):<4}], "
                progress += f"LR: {lr:.8f}, Loss: {loss_position_meter.val:.6f} (Avg: {loss_position_meter.avg:.6f}), "
                progress += f"{loss_velocity_meter.val:.6f} (Avg: {loss_velocity_meter.avg:.6f}), "
                progress += f"{loss_rotation_meter.val:.6f} (Avg: {loss_rotation_meter.avg:.6f}), "
                progress += f"Elapsed: {(end - start)*1000:.2f} ms"
                if batch_idx == len(dataloader) - 1:
                    print(progress, end='\n\r', flush=True)
                else:
                    print(progress, end="\r", flush=True)

        if accelerator.is_main_process:
            log_data = {
                'epoch': epoch + 1,
                'loss_pos': loss_position_meter.avg,
                'loss_vel': loss_velocity_meter.avg,
                'loss_rot': loss_rotation_meter.avg,
            }

    if accelerator.is_main_process:
        os.makedirs(config.train_csv_dir, exist_ok=True)
        df = pd.DataFrame([log_data])
        csv_path = os.path.join(config.train_csv_dir, 'log.csv')
        if epoch != 0 or config.use_pretrained :
            df.to_csv(csv_path, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_path, mode='w', header=True, index=False)

    else:  # Other tasks can be added here
        logger.warning(f"Task '{config.task}' is not implemented. Skipping training for this task.")




