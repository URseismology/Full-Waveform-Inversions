###############################################################################################
# Author: Sayan Kumar Swar
# Published: 05/08/2025
# University of Rochester
###############################################################################################

import jax.numpy as jnp
import numpy as np
from PIL import Image
from interpax import interp2d


def soundSpeedPhantom2D(Xi: jnp.ndarray, Yi: jnp.ndarray, image_path: str = '../data/breast_ct.jpg'):
  # read image and max normalize
  img = jnp.array(Image.open(image_path).convert('L'), dtype=np.float32)
  breastct = img / jnp.max(img)
  
  # normalize the image by performing mean subtraction
  thr = 0.04
  mask = breastct >= thr
  mean_above = jnp.mean(breastct[mask])
  breastct = jnp.where(breastct <= thr, mean_above, breastct)
  breastct = breastct - jnp.mean(breastct)
  breastct = breastct / jnp.max(jnp.abs(breastct))

  # get diemnsions of the image
  Ny, Nx = breastct.shape
  dx = dy = 0.0007
  x = jnp.linspace(-(Nx - 1) / 2 * dx, (Nx - 1) / 2 * dx, Nx)
  y = jnp.linspace(-(Ny - 1) / 2 * dy, (Ny - 1) / 2 * dy, Ny)
  X, Y = jnp.meshgrid(x, y, indexing='xy')

  # get sound speed image
  c_bkgnd = 1500.0
  c_std = 90.0
  c_map = c_bkgnd + c_std * breastct

  # rotate the grid
  R = jnp.sqrt(Xi**2 + Yi**2)
  rot_angle = 2.85 * jnp.pi
  T = jnp.arctan2(Yi, Xi) - rot_angle
  U = R * jnp.cos(T)
  V = R * jnp.sin(T)

  #C = _interp2_linear(x, y, c_map, U, V, c_bkgnd)
  
  ##!#pip install interpax
  C = interp2d(U.reshape((-1)), V.reshape((-1)), 
               x.reshape((-1)), y.reshape((-1)), 
               c_map.T, method= 'linear', extrap=c_bkgnd)
  C = C.reshape(U.shape)
  return C, c_bkgnd



#If Interpax is used then we do not need the following function
def _interp2_linear(x, y, f, xq, yq, fill_value):
    """
    Bilinear interpolation of 2D array `f` defined on grid `x, y`
    at query points (xq, yq). Points outside get `fill_value`.
    """
    Nx = x.shape[0]
    Ny = y.shape[0]
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    x0, y0 = x[0], y[0]

    # fractional index
    xi = (xq - x0) / dx
    yi = (yq - y0) / dy

    # integer part
    xi0 = jnp.floor(xi).astype(jnp.int32)
    yi0 = jnp.floor(yi).astype(jnp.int32)
    xi1 = xi0 + 1
    yi1 = yi0 + 1

    # valid mask
    valid = (
        (xi0 >= 0) & (xi1 < Nx) &
        (yi0 >= 0) & (yi1 < Ny)
    )

    # clamp indices for safe gathering
    xi0c = jnp.clip(xi0, 0, Nx - 1)
    xi1c = jnp.clip(xi1, 0, Nx - 1)
    yi0c = jnp.clip(yi0, 0, Ny - 1)
    yi1c = jnp.clip(yi1, 0, Ny - 1)

    # gather corner values
    f00 = f[yi0c, xi0c]
    f10 = f[yi0c, xi1c]
    f01 = f[yi1c, xi0c]
    f11 = f[yi1c, xi1c]

    # interpolation weights
    wx = xi - xi0
    wy = yi - yi0

    # interpolate
    f0 = f00 * (1 - wx) + f10 * wx
    f1 = f01 * (1 - wx) + f11 * wx
    fq = f0 * (1 - wy) + f1 * wy

    # apply fill for out‑of‑bounds
    return jnp.where(valid, fq, fill_value)