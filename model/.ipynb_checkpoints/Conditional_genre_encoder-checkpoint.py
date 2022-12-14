import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional

import jax
import jax.numpy as jnp
from jax import random

from flax import linen as nn
from typing import Callable, Any, Optional


class Encoder(nn.Module):
    
    linear:bool=False
    dilation:bool=False
    latent_size:int=512
    hidden_layer:int=512
    n_features:int=30
    
    @nn.compact
    def __call__(self, x,y):
        
        x = jnp.expand_dims(x, axis=-1)
        
        y = jnp.array(y, dtype='int32')
        
        y = nn.Embed(num_embeddings=30, features=512)(y)
        
        y = jnp.expand_dims(y, axis=1)
        
        y = jnp.expand_dims(y, axis=1)
        
        # x = x + y 
  

        
        # 0 
        if self.dilation:
            x = nn.Conv(512, kernel_size=(3,3),  strides=[2,2], kernel_dilation=1, padding='same')(x)
            x = x + y 
        else:
            x = nn.Conv(512, kernel_size=(3,3),  strides=[2,2], padding='same')(x)
            x = x + y
  
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        # 1
        if self.dilation:
            x = nn.Conv(512,kernel_size=(3,3), kernel_dilation=1, padding='same')(x)
        else:
            x = nn.Conv(512,kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        x = nn.max_pool(x, window_shape=(2,2), strides=(2,2))

        # 2 
        if self.dilation:
            x = nn.Conv(256,kernel_size=(3,3), kernel_dilation=2, padding='same')(x)
        else:            
            x = nn.Conv(256,kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
 
        # 3
        if self.dilation:
            x = nn.Conv(128,kernel_size=(3,3), kernel_dilation=2, padding='same')(x)
        else:
            x = nn.Conv(128,kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 4
        if self.dilation:
            x = nn.Conv(64, kernel_size=(3,3), kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(64,kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 5
        if self.dilation:
            x = nn.Conv(32, kernel_size=(3,3), kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(32, kernel_size=(3,3),  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 6
        if self.dilation:
            x = nn.Conv(16, kernel_size=(3,3), kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(16, kernel_size=(3,3), padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 7
        if self.dilation:
            x = nn.Conv(1,kernel_size=(3,3), strides=[1,1], kernel_dilation=4, padding='same')(x)
        else:
            x = nn.Conv(1,kernel_size=(3,3), strides=[1,1],  padding='same')(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)

        
        x = x.reshape(x.shape[0], -1) 
        
        
        # mean_x = nn.Dense(512, name='fc3_mean')(x)
        # logvar_x = nn.Dense(512, name='fc3_logvar')(x)  # (128, 12, 469, 20)
        
        # z = reparameterize(z_rng, mean_x, logvar_x)
        
        z = nn.Dense(features=self.latent_size, name='latent_vector')(x)
        
        if self.linear:
            z = nn.Dense(self.hidden_layer, name='linear_hidden_layer')(z)    
            z = jax.nn.leaky_relu(z) # nn.tanh(x)
            z = nn.Dense(self.n_features, name='linear_classification')(z)
        
        
        return z 
    
    
class Decoder(nn.Module):
    
    dilation:bool=False
    latent_size:int=512
    
    @nn.compact
    def __call__(self, x,y):
        
        x = nn.Dense(12 * 469 * 1)(x)
        x = x.reshape(x.shape[0], 12, 469, 1)
        
        y = jnp.array(y, dtype='int32')
        y = nn.Embed(num_embeddings=30, features=32)(y)
        
        y = jnp.expand_dims(y, axis=1)
        y = jnp.expand_dims(y, axis=1)
        
    
        # 0
        if self.dilation:
            x = nn.ConvTranspose(32, kernel_size=(3,3), strides=[1,1], kernel_dilation=(4,4))(x)
            x = x+y
       
        else:
            x = nn.ConvTranspose(32, kernel_size=(3,3), strides=[1,1])(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        # 1
        if self.dilation:
            x = nn.ConvTranspose(64, kernel_size=(3,3))(x)
        else:
            x = nn.ConvTranspose(64, kernel_size=(3,3), strides=[1,1],kernel_dilation=(2,2))(x)
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)        
        
        # 2
        if self.dilation:
            x = nn.ConvTranspose(128, kernel_size=(3,3), strides=[2,2], kernel_dilation=(2,2))(x)
        else:             
            x = nn.ConvTranspose(128, kernel_size=(3,3), strides=[2,2])(x)                   
        x = jax.nn.leaky_relu(x)
        x = nn.normalization.BatchNorm(True)(x)
        
        
        # 3
        if self.dilation:
            x = nn.ConvTranspose(256, kernel_size=(3,3), strides=[2,2], kernel_dilation=(2,2))(x)
        else:
            x = nn.ConvTranspose(256, kernel_size=(3,3), strides=[2,2])(x)
            
        x = jax.nn.leaky_relu(x)
        
        
        x = nn.ConvTranspose(1, kernel_size=(3,3), strides=[1,1])(x)
        x = jax.nn.tanh(x)
        x = jnp.squeeze(x, axis=-1)
        return x
        

class Conv2d_CAE(nn.Module):
    dilation:bool=False
    latent_size:int=512
    n_features:int=30
    
    def setup(self):
        self.encoder = Encoder(dilation=self.dilation, 
                               linear=False, 
                               latent_size=self.latent_size,
                               n_features=self.n_features,)
        self.decoder = Decoder(dilation=self.dilation, latent_size=self.latent_size)
        
        
    def __call__(self, x,y):
        
        z = self.encoder(x,y)
        recon_x = self.decoder(z,y)
        
        return recon_x    