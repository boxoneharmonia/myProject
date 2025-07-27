from config import config
import torch
import torch.nn as nn
from accelerate import Accelerator
import logging
import os

from src.net import build_model, build_criterion
from src.dataset import build_dataloader
from src.utils import build_optimizer, build_scheduler, set_all_seeds, initialize_weights
from src.trainer import train_one_epoch

logger = logging.getLogger(__name__)

def main():
    if config.use_cuda and torch.cuda.is_available():
        accelerator = Accelerator(mixed_precision=config.amp)
    else:
        accelerator = Accelerator(cpu=True)

    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO)
        logger.info(f"Starting training with {config.amp} mixed precision.")
    else:
        logging.basicConfig(level=logging.CRITICAL)

    set_all_seeds(config)
    logger.info(f"Random seed value: {config.seed}")

    os.makedirs(config.weight_dir, exist_ok=True)
    logger.info(f"Weight saved at: {config.weight_dir}")
    with open(os.path.join(config.weight_dir, 'config.txt'), 'w') as f:
        for key, value in config.__dict__.items():
            f.write(f"{key}: {value}\n")

    model = build_model(config)
    model.apply(initialize_weights)

    logger.info("Net created successfully.")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info(f"Model trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    optimizer = build_optimizer(model, config)
    logger.info(f"Optimizer: {config.optimizer}, Learning rate: {config.learning_rate}")

    if config.use_pretrained:
        weight_path = os.path.join(config.weight_dir, config.weight_name + ".pth")
        if os.path.exists(weight_path):
            logger.info(f"Loading pretrained weights from {weight_path}")
            model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
        else:
            logger.warning(f"Pretrained weights not found at {weight_path}")

    trainloader = build_dataloader(config, is_train=True)
    logger.info(f"Train dataloader created with {len(trainloader)} batches.")

    scheduler = build_scheduler(optimizer, config, len(trainloader))
    logger.info(f"{config.scheduler} scheduler created with {config.max_epochs} epochs.")

    model, optimizer, trainloader, scheduler = accelerator.prepare(
        model, optimizer, trainloader, scheduler
    )

    criterion = build_criterion(config)

    for epoch in range(config.max_epochs):
        train_one_epoch(
            model, trainloader, optimizer, scheduler, criterion, accelerator, epoch, config
        )
        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            if config.save_interval > 0 and (epoch + 1) % config.save_interval == 0:
                unwarp_model = accelerator.unwrap_model(model)
                weight_path = os.path.join(config.weight_dir, config.weight_name + ".pth")
                torch.save(unwarp_model.state_dict(), weight_path)
                logger.info(f"Model weights saved at {weight_path}")

    if accelerator.is_main_process:
        logger.info("Training completed successfully.")

    model.to('cpu')
    accelerator.end_training()
    del model, optimizer, trainloader, scheduler, criterion
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if __name__ == "__main__":
    main() 


    












