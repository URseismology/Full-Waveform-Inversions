###############################################################################################
# Author: Sayan Kumar Swar
# Published: 05/08/2025
# University of Rochester
###############################################################################################

import jax.numpy as jnp
import math


def gauspuls(t, fc, bw):
    # This Gaussian pulse code is inherited from Matlab gaupulse
    tprI = -60 
    bwrI = -6 
    bwI = bw
    fcI = fc
    r = 10**(bwrI/20)
    fv = -bwI*bwI*fcI*fcI/(8*jnp.log(r))
    tv = 1/(4 * jnp.pi * jnp.pi * fv)
    ye = jnp.exp(jnp.multiply(-1*t, t) / (2 * tv))
    xx = 2*fcI*t
    yc = ye * jnp.cos(xx * jnp.pi)
    return yc

def gauspuls_cutoff(fc, bw):   
    # This Gaussian pulse code is inherited from Matlab gaupulse
    tprI = -60 
    bwrI = -6 
    bwI = bw
    fcI = fc
    r = 10**(bwrI/20)
    fv = -bwI*bwI*fcI*fcI/(8*jnp.log(r))
    tv = 1/(4*jnp.pi*jnp.pi*fv)
    delta = 10**(tprI/20)
    yc = jnp.sqrt(-2*tv*jnp.log(delta))
    return yc



