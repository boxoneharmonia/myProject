from config import config
import torch
import logging
import os

from src.net import build_model, build_criterion
from src.dataset import build_dataloader
from src.utils import set_all_seeds
from src.valid import valid_one_epoch

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    if config.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    logger.info(f"Test on device {device}")

    set_all_seeds(config)
    logger.info(f"Random seed value: {config.seed}")

    model = build_model(config)
    weight_path = os.path.join(config.weight_dir, config.weight_name + ".pth")
    if os.path.exists(weight_path):
        logger.info(f"Loading pretrained weights from {weight_path}")
        model.load_state_dict(torch.load(weight_path, map_location='cpu'), strict=False)
    else:
        logger.warning(f"Pretrained weights not found at {weight_path}")
        exit(0)    

    testloader = build_dataloader(config, is_train=False)
    logger.info(f"Test dataloader created with {len(testloader)} batches.")

    criterion = build_criterion(config)
    model.eval()
    model.to(device)
    valid_one_epoch(model, testloader, criterion, device, config)

    logger.info("Test completed successfully.")

    model.to('cpu')
    del model, testloader, criterion
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

if __name__ == "__main__":
    main() 