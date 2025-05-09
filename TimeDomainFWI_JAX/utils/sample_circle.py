###############################################################################################
# Author: Sayan Kumar Swar
# Published: 05/08/2025
# University of Rochester
###############################################################################################


import jax.numpy as jnp
from scipy.spatial import cKDTree

def sampled_circle(Nx, Nz, Nr, theta):

    x = jnp.arange(-(Nx - 1) / 2, (Nx - 1) / 2 + 1)
    z = jnp.arange(-(Nz - 1) / 2, (Nz - 1) / 2 + 1)

    # Target 
    x_target = Nr * jnp.cos(theta)
    z_target = Nr * jnp.sin(theta)

    # nearest neighbor search
    tree_x = cKDTree(x[:, None])
    tree_z = cKDTree(z[:, None])

    x_idx = tree_x.query(x_target[:, None])[1]
    z_idx = tree_z.query(z_target[:, None])[1]

    ind = z_idx * Nx + x_idx

    return x_idx, z_idx, ind
