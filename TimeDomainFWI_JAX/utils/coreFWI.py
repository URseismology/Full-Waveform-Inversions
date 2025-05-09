###############################################################################################
# Author: Sayan Kumar Swar
# Published: 05/08/2025
# University of Rochester
###############################################################################################

import jax
import jax.numpy as jnp
from jax import lax
from jaxopt.tree_util import tree_add_scalar_mul
from jaxopt.tree_util import tree_div
from jaxopt.tree_util import tree_l2_norm
from jaxopt.tree_util import tree_scalar_mul
from jaxopt.tree_util import tree_sub
from jaxopt.tree_util import tree_vdot_real
from jaxopt.tree_util import tree_conj

import matplotlib.pyplot as plt 
jax.config.update("jax_enable_x64", True)


def elementwise_multiply(x, y):
  return x * y
  # apply vmap to batch multiply the forward operation
batch_multiply = jax.vmap(elementwise_multiply)


def getPerfectMatchingLayer(X,Y,Nx,Ny,dx,dy,dt,a0,L_PML):
    wtotalx = (Nx - 1) * dx
    wintx = wtotalx - 2 * L_PML

    wtotaly = (Ny - 1) * dy
    winty = wtotaly - 2 * L_PML

    PMLx = 2 * jnp.pi * a0 * ((jnp.maximum(jnp.abs(X) - wintx / 2, 0) / L_PML) ** 2) / dt
    PMLy = 2 * jnp.pi * a0 * ((jnp.maximum(jnp.abs(Y) - winty / 2, 0) / L_PML) ** 2) / dt

    PML = PMLx + PMLy

    return PML

def getLaplacianSpatialDervatives(Nx,Ny,dx,dy,dt,vel):
    # K-space spectrum method
    kx = 2 * jnp.pi * (jnp.mod((jnp.arange(0,Nx)/(dx * Nx) + 1 / (2 * dx)), 1/dx) - 1/( 2 * dx))
    ky = 2 * jnp.pi * (jnp.mod(jnp.arange(Ny) / (dy * Ny) + 1 / (2 * dy), 1 / dy) - 1 / (2 * dy))
    Ky, Kx = jnp.meshgrid(ky, kx, indexing='xy') 
    K = jnp.sqrt(Kx**2 + Ky**2)

    cref = jnp.mean(vel.flatten())
    kappa = jnp.sinc(cref * K * dt / (2 * jnp.pi))
    Laplacian = - (kappa * K) ** 2
    return Laplacian


def applyLaplacian(Laplacian, img=jnp.ndarray) -> jnp.ndarray:
    # This function is to be used with Jax ForI loop & the Naive wavefield forward computation
    img_k = jnp.fft.fft2(img, axes=(0,1))
    out_k = batch_multiply(Laplacian[...,None], img_k)
    L = jnp.real(jnp.fft.ifft2(out_k, axes=(0,1)))
    return L


def getTimeStepping_naive(x, y, vel, src, t, a0, L_PML, adjoint=0, isplot=0):
    # This function is written to build understanding of time stepping finite differencing technique
    # This function is highly inefficient with Jax but is solid foundation of theory 
    # And to formulate Jax optimized code later
    
    Nsrcs = src.shape[3]
    dt = jnp.mean(jnp.diff(t)).astype(jnp.float32)
    dx = jnp.mean(jnp.diff(x)).astype(jnp.float32)
    dy = jnp.mean(jnp.diff(y)).astype(jnp.float32)
    Nt = t.shape[0]
    Nx = x.shape[0]
    Ny = y.shape[0]
    X, Y = jnp.meshgrid(x, y, indexing='xy')
    
    print(dx,dy,dt)
    
    # get the PML damping and the spatial derivative function
    PML = getPerfectMatchingLayer(X, Y, Nx, Ny, dx, dy, a0, L_PML)
    laplacian = getLaplacianSpatialDervatives(Nx, Ny, dx, dy, dt, vel)

    # get wavenumber sqaured for easier computation
    lambda_sq = (vel*dt)**2; 

    # plot the visulas of the laplacian, PML operators for Sanity
    if isplot == 1:
        plt.figure(figsize=(8,4))
        plt.subplot(1,3,1)
        plt.imshow(laplacian, cmap='viridis')
        plt.title('Laplacian Operator')

        plt.subplot(1,3,2)
        plt.imshow(PML, cmap='viridis')
        plt.title('PML Operator')

        plt.subplot(1,3,3)
        plt.imshow(lambda_sq, cmap='viridis')
        plt.title('lambda_sq Operator')

        plt.show()

        print(laplacian.max(),laplacian.min())
        print(PML.max(),PML.min())
    
    
    # get the forward time step
    wvfield = jnp.zeros([Ny, Nx, Nt, Nsrcs])
    if adjoint==0:
        for n in range(Nt):
            if n == 0:
                damping = (1 + PML * dt)
                #u_next = ((-src[:, :, n, :]) * (lambda_sq)) / (damping)
                u_next = jnp.multiply(-src[:, :, n, :], lambda_sq[..., None]) / damping[..., None]
            
            elif n == 1:
                u_prev = wvfield[:, :, n-1, :]
                u_curr = wvfield[:, :, n-1, :] 
                L = applyLaplacian(laplacian,u_curr)
                damping_denom = (1 + PML * dt)
                damping_nuemo = (2 - (PML * dt) ** 2)
                
                u_next = (jnp.multiply(damping_nuemo[...,None],u_curr) 
                            + jnp.multiply((L - src[:, :, n, :]), lambda_sq[..., None])
                        ) / damping_denom[..., None]
                
                ## Old Code
                # u_next = (
                #     (2 - (PML * dt) ** 2) * u_curr
                #     + (L - src[:, :, n, :]) * lambda_sq[..., None]
                # ) / (1 + PML * dt)
            
            else:
                u_nm1 = wvfield[:, :, n-1, :]
                u_nm2 = wvfield[:, :, n-2, :]
                L = applyLaplacian(laplacian,u_nm1)
                damping_denom = (1 + PML * dt)
                damping_nuemoA = (2 - (PML * dt) ** 2)
                damping_nuemoB = (1 - (PML * dt))
                u_next = (
                    jnp.multiply(damping_nuemoA[...,None],u_nm1) 
                    - jnp.multiply(damping_nuemoB[...,None],u_nm2)
                    + jnp.multiply((L - src[:, :, n, :]),lambda_sq[..., None])
                ) / damping_denom[..., None]

                ## Old Code
                # u_next = (
                #     (2 - (PML * dt) ** 2) * u_nm1
                #     - (1 - PML * dt) * u_nm2
                #     + (L - src[:, :, n, :]) * lambda_sq[..., None]
                # ) / (1 + PML * dt)

            wvfield = wvfield.at[:, :, n, :].set(u_next)

            # if n==0:
            #     break
    else:
        pass

    return wvfield

@jax.jit
def getTimeStepping_vec(x, y, vel, src, t, a0, L_PML, adjoint=0):
    # grid/time sizes & spacings
    Nt = t.shape[0]
    Ny, Nx = y.size, x.size
    Nsrcs = src.shape[3]
    dt = jnp.mean(jnp.diff(t))
    
    # build PML & Laplacian operator once
    X, Y = jnp.meshgrid(x, y, indexing='xy')
    PML = getPerfectMatchingLayer(X, Y, Nx, Ny, jnp.mean(jnp.diff(x)), jnp.mean(jnp.diff(y)),dt, a0, L_PML)
    lap_op = getLaplacianSpatialDervatives(Nx, Ny, jnp.mean(jnp.diff(x)), jnp.mean(jnp.diff(y)), dt, vel)
    lap_op = lap_op[..., None]
    
    # apply laplacian on current time step of the wavefiled propagation
    def applyL(u: jnp.ndarray) -> jnp.ndarray:
        U_k = jnp.fft.fft2(u, axes=(0,1))
        return jnp.real(jnp.fft.ifft2(lap_op * U_k, axes=(0,1)))
    
    
    lam2 = (vel * dt)**2      #dim: Ny-Nx
    lam2 = lam2[..., None]    #dim: Ny-Nx-1
    damp1 = 1.0 + PML * dt    #dim: Ny-Nx
    damp1 = damp1[..., None]  #dim: Ny-Nx-1
    dt2   = (PML * dt)**2     #dim: Ny-Nx
    dt2   = dt2[..., None]
    
    # reorder src to time‐major to iterate over the time axis
    src_t = jnp.transpose(src, (2, 0, 1, 3))                            #dim: Nt-Ny-Nx-Nsrcs
    times = jnp.arange(Nt, dtype=jnp.int32)

    # the follwing function will step through each time step and apply the finite differencing method
    def step(carry, inp):
        u_nm1, u_nm2 = carry                                            #dim: Ny-Nx-Nsrcs
        n, s = inp                                                      #dim n: enumeration index; dim s: Ny-Nx-Nsrcs

        #get the finite differece solution on case basis for indices
        def case0(_):
            return (-s * lam2) / damp1

        def case1(_):
            L1 = applyL(u_nm1)
            num = (2 - dt2) * u_nm1 + (L1 - s) * lam2
            return num / damp1

        def case_else(_):
            L1 = applyL(u_nm1)
            num = (2 - dt2) * u_nm1 \
                  - (1 - PML*dt)[..., None] * u_nm2 \
                  + (L1 - s) * lam2
            return num / damp1

        branches = [case0, case1, case_else]
        u_next = lax.switch(n, branches, None)
        return (u_next, u_nm1), u_next

    init = (jnp.zeros((Ny, Nx, Nsrcs)), jnp.zeros((Ny, Nx, Nsrcs)))
    (_, _), wv_t = lax.scan(step, init, (times, src_t))
    
    # transpose back to Ny-Nx-Nt-Nsrcs
    return jnp.transpose(wv_t, (1, 2, 0, 3))


@jax.jit
def getTimeStepping_fori(x, y, vel, src, t, a0, L_PML,adjoint=0):
    Nsrcs = src.shape[3]
    dt = jnp.mean(jnp.diff(t))
    dx = jnp.mean(jnp.diff(x))
    dy = jnp.mean(jnp.diff(y))
    Nt = t.shape[0]
    Nx = x.shape[0]
    Ny = y.shape[0]
    X, Y = jnp.meshgrid(x, y, indexing='xy')

    # get the PML damping and the spatial derivative function
    PML = getPerfectMatchingLayer(X, Y, Nx, Ny, dx, dy, dt, a0, L_PML)
    laplacian = getLaplacianSpatialDervatives(Nx, Ny, dx, dy, dt, vel)

    # get wavenumber sqaured for easier computation
    lambda_sq = (vel*dt)**2; 

    # compute all the static variables
    pml_damping_denom = (1 + PML * dt)
    pml_damping_nuemoA = (2 - (PML * dt) ** 2)
    pml_damping_nuemoB = (1 - (PML * dt))

    wvfield = jnp.zeros([Ny, Nx, Nt, Nsrcs])
    
    # get the finite differece solution for the initial index
    def compute_uinit(tidx,u):

        return jax.lax.cond(tidx==0,
            lambda: batch_multiply(-src[:, :, tidx, :], lambda_sq[..., None]) / pml_damping_denom[..., None],
            lambda: compute_unext(tidx,u)
        )

    # get the finite differece solution for the initial index
    def compute_unext(tidx,u):

        return jax.lax.cond(tidx==1,
            lambda: (batch_multiply(pml_damping_nuemoA[...,None],u[:, :, tidx-1, :]) 
                                + batch_multiply((applyLaplacian(laplacian,u[:, :, tidx-1, :]) - src[:, :, tidx, :]), lambda_sq[..., None])
                            ) / pml_damping_denom[..., None],

            lambda: (batch_multiply(pml_damping_nuemoA[...,None],u[:, :, tidx-1, :]) 
                    - batch_multiply(pml_damping_nuemoB[...,None],u[:, :, tidx-2, :])
                    + batch_multiply((applyLaplacian(laplacian,u[:, :, tidx-1, :]) - src[:, :, tidx, :]),lambda_sq[..., None])
                ) / pml_damping_denom[..., None]
        )
    
    def update_uall(tidx,u):
        unxt = compute_uinit(tidx,u)
        return u.at[:, :, tidx, :].set(unxt)                            
    
    wvfield = jax.lax.fori_loop(0,Nt,update_uall,wvfield)               #dim: Ny-Nx-Nt-Nsrcs OR Axial-Lateral-Timeseries-Transmitter
        
    return wvfield



## Hessian-Vector Product - Forward-Over-Reverse
## The Following Code is written by Dr. Rehman Ali
def hvp(f, primals, tangents):
    return jax.jvp(jax.grad(f), primals, tangents)[1]


## Linearized CG with Closed-Form Step Size Calculation
## Conjugate Gradient Method (with Exact Line Search Step Size Calculation)
## The Following Code is written by Dr. Rehman Ali
class LinearizedCG:
    def __init__(self, fun, maxiter=500, method="polak-ribiere"):
        self.fun = fun; 
        self.maxiter = maxiter; 
        self.method = method; 
        self.iter = 0; 
    def init_state(self, params, *args, **kwargs):
        # Initial input params
        #pdb.set_trace();
        self.params = params; 
        # Objective Function with *args, **kwargs 
        self.obj_fun = lambda params: self.fun(params, *args, **kwargs); 
    def update(self):
        # Compute Gradient
        new_grad = jax.grad(self.obj_fun)(self.params); 
        if self.iter == 0:
            # Calculate Search Direction
            descent_direction = tree_scalar_mul(-1, new_grad)
        else:
            # Load Previous Gradient and Descent Direction
            grad = self.grad; 
            descent_direction = self.descent_direction; 
            # Calculate Momentum
            eps = 1e-16
            if self.method == "polak-ribiere":
                # See Numerical Optimization, second edition, equation (5.44).
                gTg = tree_vdot_real(grad, grad)
                gTg = jnp.where(gTg >= eps, gTg, eps)
                momentum = tree_vdot_real(
                    tree_conj(tree_sub(new_grad, grad)), tree_conj(new_grad)) / gTg
                momentum = jax.nn.relu(momentum)
            elif self.method == "fletcher-reeves":
                # See Numerical Optimization, second edition, equation (5.41a).
                gTg = tree_vdot_real(grad, grad)
                gTg = jnp.where(gTg >= eps, gTg, eps)
                momentum = tree_vdot_real(new_grad, new_grad) / gTg
                momentum = jax.nn.relu(momentum)
            elif self.method == "hestenes-stiefel":
                # See Numerical Optimization, second edition, equation (5.45).
                grad_diff = tree_sub(new_grad, grad)
                dTg = tree_vdot_real(tree_conj(grad_diff), descent_direction)
                dTg = jnp.where(dTg >= eps, dTg, eps)
                momentum = tree_vdot_real(
                    tree_conj(grad_diff), tree_conj(new_grad)) / dTg
                momentum = jax.nn.relu(momentum)
            else:
                raise ValueError("method argument should be either 'polak-ribiere', "
                            "'fletcher-reeves', or 'hestenes-stiefel'.")
            # Calculate Search Direction
            descent_direction = tree_sub(tree_scalar_mul(momentum, descent_direction), new_grad);
        # Step Size
        stepsize = (tree_vdot_real(new_grad, descent_direction) / \
            tree_vdot_real(descent_direction, hvp(self.obj_fun, (self.params,), (descent_direction,))))
        # 6) Update
        self.params = tree_add_scalar_mul(self.params, -stepsize, descent_direction)
        # Record Previous Gradient
        self.grad = new_grad
        self.descent_direction = descent_direction
        self.iter += 1
        return self.params
    def run(self, params, *args, **kwargs):
        # Initialize Run State
        self.init_state(params, *args, **kwargs); 
        # Loop Until Max Iterations Hit
        while self.iter < self.maxiter:
            self.update(); 
        return self.params; 