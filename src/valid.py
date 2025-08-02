import os
import time
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from .utils import AverageMeter

logger = logging.getLogger(__name__)

def valid_one_epoch(model, dataloader, criterion, device, config):
    model.eval()
    model.to(device)
    log_data = []

    if config.task == 'mlm':
        loss_meter = AverageMeter('-')

        for batch_idx, x_seq in enumerate(dataloader):
            start = time.time()
            x_seq = x_seq.to(device)
            
            with torch.no_grad():
                output = model(x_seq, config.mask_probability)
                loss = criterion(output)

            end = time.time()
            loss_meter.update(loss.item())
            progress = f"Batch: [{batch_idx+1:>4}/{len(dataloader):<4}], "
            progress += f"Loss: {loss_meter.val:.6f} (Avg: {loss_meter.avg:.6f}), "
            progress += f"Elapsed: {(end - start)*1000:.2f} ms"

            if batch_idx == len(dataloader) - 1:
                print(progress, end='\n\r', flush=True)
            else:
                print(progress, end='\r', flush=True)

            if batch_idx % (len(dataloader)//8) == 0:
                x_gt, x_pred = output
                x_gt = x_gt.detach().cpu()    # (B, S, C, H, W)
                x_pred = x_pred.detach().cpu()  # (B, S, C, H, W)

                sample_gt = x_gt[0]   # (S, C, H, W)
                sample_pred = x_pred[0] # (S, C, H, W)

                true_pos_seq = sample_gt[:, 0, :, :] # (S, H, W)
                pred_pos_seq = sample_pred[:, 0, :, :] # (S, H, W)
                target_size = pred_pos_seq.shape[-2:]

                for i in range(2):
                    true_pos_img = F.interpolate(true_pos_seq[i].unsqueeze(0).unsqueeze(0), size=target_size, mode='bilinear').squeeze(0)
                    plt.subplot(1, 2, 1)
                    plt.imshow(tensor_to_numpy(true_pos_img), cmap='gray')
                    plt.title("true pos img")

                    plt.subplot(1, 2, 2)
                    plt.imshow(tensor_to_numpy(pred_pos_seq[i].unsqueeze(0)), cmap='gray')
                    plt.title("pred pos img")
                    plt.show()

            log_data.append({'batch_idx': batch_idx + 1, 'loss': loss_meter.val})

        os.makedirs(config.valid_csv_dir, exist_ok=True)
        csv_path = os.path.join(config.valid_csv_dir, 'log.csv')
        df = pd.DataFrame(log_data)
        df.to_csv(csv_path, mode='w', header=True, index=False)

        logger.info(f"Validation finished. Average Loss: {loss_meter.avg:.6f}")

def tensor_to_numpy(tensor:torch.Tensor):
    """
    Args:
        tensor (torch.Tensor): (C, H, W)
    Returns:
        np.ndarray: (H, W, C)
    """
    if tensor.ndim == 4:
        tensor = tensor[0]

    denormalized_tensor = tensor * 0.5 + 0.5
    scaled_tensor = denormalized_tensor * 255
    clamped_tensor = torch.clamp(scaled_tensor, 0, 255)
    permuted_tensor = clamped_tensor.permute(1, 2, 0)
    image = permuted_tensor.to(torch.uint8).cpu().numpy()

    return image            

