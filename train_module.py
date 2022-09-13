# 2022-09-02 16:23 Seoul

# --- import dataset ---
from utils.losses import *
from torch.utils.data import DataLoader, random_split

# --- import model ---
from model.Conv2d_model import Conv2d_VAE, Encoder
from model.Conditional_genre_encoder import Conv2d_CAE

# --- import framework ---
import flax 
from flax import jax_utils
from flax.training import train_state, common_utils, checkpoints
import jax
import numpy as np
import jax.numpy as jnp
import optax

from tqdm import tqdm
import os
import wandb
from utils.config_hook import yaml_config_hook
import matplotlib.pyplot as plt

from functools import partial

# --- Define config ---
config_dir = os.path.join(os.path.expanduser('~'),'module/config')     
config = yaml_config_hook(os.path.join(config_dir, 'config.yaml'))

class TrainerModule:

    def __init__(self, 
                 seed,
                 config):
        
        super().__init__()
        self.config = config
        self.seed = seed
        self.exmp = jnp.ones((self.config['batch_size'], 48, 1876))
        # Create empty model. Note: no parameters yet
        self.model = Conv2d_CAE(dilation=self.config['dilation'], latent_size=self.config['latent_size'])
        # self.linear_model = linear_evaluation()
        self.Encoder = Encoder(dilation=self.config['dilation'])
        # Prepare logging
        self.log_dir = self.config['checkpoints_path']
        # Create jitted training and eval functions
        self.create_functions()
        # Initialize model
        self.init_model()
        wandb.init(
        project=config['model_type'],
        entity='aiffelthon',
        config=config
        )
        
    def create_functions(self):
        # Training function
        def train_step(state, batch):
            
            def loss_fn(params):
                mel, y = batch
                mel = (mel/200) + 0.5
                recon_x = self.model.apply(params, mel, y)
                loss = ((recon_x - mel) ** 2).mean()
                return loss
            
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
            grads = jax.lax.pmean(grads, axis_name='batch')
            state = state.apply_gradients(grads=grads)  # Optimizer update step
            
            return state, loss
        self.train_step = jax.pmap(partial(train_step), axis_name='batch')
        
        # Eval function        
        def eval_step(state, batch):
            mel, y = batch
            mel = (mel/200) + 0.5
            recon_x = self.model.apply(state.params, mel, y)
            loss = ((recon_x - mel) ** 2).mean()
            
            return loss, recon_x        
        self.eval_step = jax.pmap(partial(eval_step), axis_name='batch')
        
        
    def create_png(test_x, recon):
        recon = jax_utils.unreplicate(recon)
        fig1, ax1 = plt.subplots()
        im1 = ax1.imshow(recon_x[0], aspect='auto', origin='lower', interpolation='none')
        fig1.colorbar(im1)
        fig1.savefig('recon.png')
        plt.close(fig1)

        fig2, ax2 = plt.subplots()
        im2 = ax2.imshow(test_x[0], aspect='auto', origin='lower', interpolation='none')
        fig2.colorbar(im2)
        fig2.savefig('x.png')
        plt.close(fig2)
        
    def init_model(self):
        # Initialize model
        rng = jax.random.PRNGKey(self.seed)
        rng, init_rng = jax.random.split(rng)
        params = self.model.init(init_rng, self.exmp, jnp.ones((self.config['batch_size'])))
        # Initialize optimizer
        optimizer = optax.adam(self.config['learning_rate'])
        
        # Initialize training state
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=optimizer)
        
#     def init_linear_model(self):
#         # Initialize model
#         rng = jax.random.PRNGKey(self.seed)
#         rng, init_rng = jax.random.split(rng)
#         params = self.model.init(init_rng, jnp.ones((self.config['batch_size'], self.config['latent_size'])))
#         return params
        


    def train_model(self, train_dataloader, test_dataloader, num_epochs=5):
        # Train model for defined number of epochs
        
        self.state = jax_utils.replicate(self.state)
        for epoch_idx in range(1, num_epochs+1):
            self.train_epoch(epoch_idx, train_dataloader, test_dataloader)

        self.state = jax_utils.unreplicate(self.state)
        
    def train_epoch(self, epoch, train_dataloader, test_dataloader):
        train_dataiter = iter(train_dataloader)
        test_dataiter = iter(test_dataloader)
        
        for batch in tqdm(range(len(train_dataloader)-1), desc=f'Epoch {epoch}'):
            train_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(train_dataiter)))
            test_batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, next(test_dataiter)))
            
            self.state, train_loss = self.train_step(self.state, train_batch)
            test_loss, recon_x = self.eval_step(self.state, test_batch)

            wandb.log({'train_loss': jax.device_get(train_loss.mean()), 'test_loss': jax.device_get(test_loss.mean())})
            
            if self.state.step[0] % 100 == 0:
                recon_x = jax_utils.unreplicate(recon_x)
                fig1, ax1 = plt.subplots()
                im1 = ax1.imshow(recon_x[0], aspect='auto', origin='lower', interpolation='none')
                fig1.colorbar(im1)
                fig1.savefig('recon.png')
                plt.close(fig1)
                
                test_x = jax_utils.unreplicate(test_batch[0])
                fig2, ax2 = plt.subplots()
                im2 = ax2.imshow(test_x[0], aspect='auto', origin='lower', interpolation='none')
                fig2.colorbar(im2)
                fig2.savefig('x.png')
                plt.close(fig2)
                
                wandb.log({'reconstruction' : [
                            wandb.Image('recon.png')
                            ], 
                           'original image' : [
                            wandb.Image('x.png')
                            ]})
                

    def save_model(self, checkpoint_name, step=0):
        # Save current model at certain training iteration
        checkpoints.save_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f"genre_encoder_{checkpoint_name}_{config['latent_size']}_", step=step)
        

    def load_model(self, pretrained=False):
        # Load model. We use different checkpoint for pretrained models
        if not pretrained:
            params = checkpoints.restore_checkpoint(ckpt_dir=self.log_dir, target=self.state.params, prefix=f"{config['projector_target']}_{config['latent_size']}")
        else:
            params = checkpoints.restore_checkpoint(ckpt_dir=os.path.join(CHECKPOINT_PATH, f"{config['projector_target']}_{config['latent_size']}.ckpt"), target=self.state.params)
        self.state = train_state.TrainState.create(apply_fn=self.model.apply, params=params, tx=self.state.tx)

    def checkpoint_exists(self):
        # Check whether a pretrained model exist for this autoencoder
        return os.path.isfile(os.path.join(CHECKPOINT_PATH, f"{config['projector_target']}_{config['latent_size']}.ckpt"))

    
#     def linear_evalutaion(self, freeze_encoder=True):
                  
#         if freeze_encoder:
#             enc_state = self.state.params['params']['encoder']
#             enc_batch = self.state.params['batch_stats']['encoder']
#             linear_state = init_state(linear_evaluation(), (config['batch_size'], config['latent_size']), rng, config['learning_rate'])

        






