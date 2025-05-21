import sys
if './' not in sys.path:
	sys.path.append('./')
	
from omegaconf import OmegaConf
import argparse
import torch
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from ldm.util import instantiate_from_config
from models.util import load_state_dict
from models.logger import ImageLogger


parser = argparse.ArgumentParser(description='Uni-ControlNet Training')
parser.add_argument('--config-path', type=str, default='./configs/local_v15.yaml')
parser.add_argument('--learning-rate', type=float, default=1e-5)
parser.add_argument('---batch-size', type=int, default=4)
parser.add_argument('---training-steps', type=int, default=1e5)
parser.add_argument('---resume-path', type=str, default='./ckpt/init_local.ckpt')
parser.add_argument('---logdir', type=str, default='./log_local/')
parser.add_argument('---log-freq', type=int, default=500)
parser.add_argument('---sd-locked', type=bool, default=True)
parser.add_argument('---num-workers', type=int, default=4)
parser.add_argument('---gpus', type=int, default=-1)
parser.add_argument("---max-epochs", type=int, default=2)
parser.add_argument("--checkpoint-dirpath", type=str, default='checkpoints/vimeo_all/')  
args = parser.parse_args()


def main():

    config_path = args.config_path
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    training_steps = args.training_steps
    resume_path = args.resume_path
    default_logdir = args.logdir
    logger_freq = args.log_freq
    sd_locked = args.sd_locked
    num_workers = args.num_workers
    gpus = args.gpus

    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config['model'])

    state_dict = torch.load(resume_path, map_location='cpu',
                            # weights_only=True
                             )
    if 'state_dict' in state_dict:
        model.load_state_dict(state_dict['state_dict'])
    else:
        model.load_state_dict(state_dict)
        
    model.learning_rate = learning_rate
    model.sd_locked = sd_locked

    dataset = instantiate_from_config(config['data'])

    total_size = len(dataset)
    train_size = int(0.3 * total_size)
    indices = list(range(total_size))

    # Split the dataset into a training subset
    train_indices, _ = train_test_split(indices, train_size=train_size, random_state=42)

    # Create the training subset
    train_subset = Subset(dataset, train_indices)
    
    dataloader = DataLoader(train_subset, 
                            num_workers=num_workers, 
                            batch_size=batch_size,
                            pin_memory=True, 
                            shuffle=True, 
                            persistent_workers=True)

    #logger = ImageLogger(batch_frequency=logger_freq,num_local_conditions=2)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=logger_freq,
        dirpath= args.checkpoint_dirpath,
        filename='local-best-checkpoint'
    )
        
    trainer = pl.Trainer(
        accumulate_grad_batches=4,
        logger=True,   
        gpus=gpus,
        callbacks=[checkpoint_callback], 
        default_root_dir=default_logdir,
        max_steps=training_steps,
        detect_anomaly=True,
    )
    trainer.fit(model,
        dataloader, 
    )


if __name__ == '__main__':
    main()