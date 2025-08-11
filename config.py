import logging

logger = logging.getLogger(__name__)

class Config:
    def __init__(self):
        # Conv configuration 
        self.conv_params = [
            [32, 128, 2, 2.0], 
            [128, 128, 1, 2.0],   
            [128, 512, 2, 2.0], 
            [512, 512, 1, 2.0], 
            [512, 1024, 2, 2.0],
            [1024, 1024, 1, 2.0], 
        ]
        self.deconv_params = [
            [768, 512],   
            [512, 128], 
        ]
        # self.patch_size = (3,3)
        self.patches = 64

        # Transformer configuration 
        self.max_seq_len = 16
        self.embed_dim = 768
        self.depth = 12 
        self.depth_head = 4
        self.num_heads = 16
        self.mlp_ratio = 4.0 
        self.qkv_bias = True
        self.qk_scale = None
        self.drop_ratio = 0.0
        self.attn_drop_ratio = 0.0
        self.drop_path_ratio = 0.0 
        
        # Dataset configuration
        self.train_root = './dataset/trainv2'
        self.pretrain_root = './dataset/pretrainv2'
        self.test_root = './dataset/validv2'
        self.batch_size = 3
        self.num_workers = 2

        # Optimizer configuration
        self.optimizer = 'adamw'
        self.learning_rate = 1e-5
        self.weight_decay = 0.01
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.99
        self.momentum = 0.9
        self.max_iter = 10
        self.history_size = 20

        #Scheduler configuration
        self.scheduler = 'linear' 
        # huggingface scheduler: "linear", "cosine", "cosine_with_restarts", "polynomial", "constant", 
        # "constant_with_warmup", "inverse_sqrt", "cosine_with_min_lr"
        self.max_epochs = 5
        self.warmup_proportion = 0.1
        self.num_cycles = 0.5 # Number of cycles for cosine/cosine_with_min_lr scheduler (float) or for cosine_with_restarts scheduler (int), must be careful with this parameter
        self.power = 1.0 # Polynomial decay power for polynomial scheduler
        self.min_lr_rate = 1e-8 # Minimum learning rate for cosine_with_min_lr scheduler (not a ratio)

        # Training configuration
        self.task = 'mlm'  # Masked Language Modeling
        self.mask_probability = 0.2
        self.max_grad_norm = 1.0
        self.train_csv_dir = './log/train'
        self.valid_csv_dir = './log/valid'
        self.freeze = True

        self.alpha_pos = 1.0
        self.alpha_vel = 5.0
        self.alpha_rot = 1.0

        self.use_cuda = True
        self.amp = 'bf16' # accelerator mixed precision: 'no', 'fp16', 'bf16'
        self.accumulate = 1
        self.seed = 19260817
        self.weight_dir = './weight'
        self.weight_name = 'event_bert_mlm'
        self.use_pretrained = False
        self.save_interval = 5  # Save model every n epochs
        

    def update(self, conv_config=None, transformer_config=None, dataset_config=None,
               optimizer_config=None, scheduler_config=None, training_config=None):
        
        config_map = {
            'conv_config': conv_config,
            'transformer_config': transformer_config,
            'dataset_config': dataset_config,
            'optimizer_config': optimizer_config,
            'scheduler_config': scheduler_config,
            'training_config': training_config
        }

        for config_name, config_dict in config_map.items():
            if config_dict:
                for key, value in config_dict.items():
                    if hasattr(self, key):
                        setattr(self, key, value)
                        # logger.info(f"set {key} as {value}.")
                    else:
                        logger.warning(f"'{key}' not exist!")

config = Config()

transformer_config = {
    'max_seq_len'   : 6,
}

dataset_config = {
    'batch_size'    : 96,
    'num_workers'   : 6,
}
optimizer_config = {
    'max_epochs'        : 30,
    'warmup_proportion' : 0.05,
}

scheduler_config = {
    'scheduler'     :'polynomial',
    'learning_rate' : 5e-5,
    'weight_decay'  : 0.01,
    'min_lr_rate'   : 1e-8,
    'power'         : 2.0,
}

training_config = {
    'task'              : 'traj_v2',
    'freeze'            : True,
    'use_pretrained'    : False,
    'mask_probability'  : 0.5,
    'accumulate'        : 2,
    'max_grad_norm'     : 1.0,
    'save_interval'     : 5,
    'weight_name'       : 'event_bert_v2',
}

config.update(transformer_config=transformer_config, 
              dataset_config=dataset_config,
              optimizer_config=optimizer_config, 
              scheduler_config=scheduler_config, 
              training_config=training_config)

