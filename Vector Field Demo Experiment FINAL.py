# -*- coding: utf-8 -*-
"""
Created on Mon May  8 14:03:34 2023

@author: vaugh
"""
import numpy as np
import matplotlib.pyplot as plt

import multiplicationoperators as mo

# %% Set Parameters

# data generation parameters
num_pts = 3000
dim = 3

#plotting point size
ms = .75

# kernel bandwidth parameter
eps = .3

# tau reconstruction parameter
tau = 0

# maximum number of eigenvalues and eigenvectors to generate
num_eigs = 200

# number of eigenfunctions to use for SEC approximation
I = 50
J = 20
K = 20

# %% Generate Data

X = np.random.randn(dim,num_pts)
X /= np.linalg.norm(X,axis=0)


# %% Generate Vector Field Note this is circular

def test_func(x):
    output = np.exp(x[1] ** 2 )
    return output

v = mo.spherical_grad(X,test_func,eps)

# %% Generate Laplacian Eigenvalues and Eigenfunctions Using Diffusion Maps

S,eigvals,eigvecs,aff = mo.gen_dm_laplacian_updated(X,eps,num_eigs)

# %% Compute the Embedding Function Fourier Coefficients F_ak
F_ak_dm = (1 / num_pts) *  (X @ eigvecs[:,0:I])

# %% Generate Structure Coefficients c_ijk

# Helper function for generating c_ijk
def c_func_mc_dm(i, j, k, eigvecs):
    num_pts,num_eigs = eigvecs.shape
    return (1 / num_pts) * np.sum( eigvecs [:, i] * eigvecs [:, j] * eigvecs[:, k])

# Generate an array of all cijk's from i = 0 to i = I
c_mc_dm = [c_func_mc_dm(i, j, k, eigvecs[:,0:I]) for i in range(0, I)
           for j in range(0, I) 
           for k in range(0, I)]
            
c_mc_dm = np.reshape(np.array(c_mc_dm), (I, I, I ))

# %% Compute Riemannian Metric Coefficients g_ijk
# 
g_mc_dm = np.empty([I, I, I], dtype = float)
for i in range(0, I):
            for j in range(0, I):
                        for k in range(0, I):
                                    g_mc_dm[i,j,k] = (eigvals[i] + eigvals[j] - eigvals[k]) * c_mc_dm[i,j,k] / 2

# %% Compute Entrees of Gram Operator G_ijpq 

G_mc_dm = np.zeros([I, I, I, I], dtype = float)
G_mc_dm = np.einsum('ipm, jqm -> ijpq', c_mc_dm, g_mc_dm, dtype = float)

G_mc_dm = G_mc_dm[:J, :K, :J, :K]
G_mc_dm = np.reshape(G_mc_dm, (J*K, J*K))

# %% Compute The Dual Gram Operator G_dual_ijpq (using pseudoinverse)

# Threshold value for truncated SVD
threshold = 1/28
G_dual_mc_dm = np.linalg.pinv(G_mc_dm, rcond = threshold)

# L2 integral of products between eigenfunction phi_mn and "arrows" v_an
def monte_carlo_product_dm(eigvecs, v):
    dim,num_pts = X.shape
    integral = (1 / num_pts) * np.sum(eigvecs * v, axis = 1)
    
    return integral

# %% Compute b_am entries using (L2) deterministic Monte Carlo integral

def b_func_mc_dm(i):
    return monte_carlo_product_dm(eigvecs[:, i], v)

b_am_mc_dm =  [b_func_mc_dm(m) for m in range(I)]
b_am_mc_dm = np.array(b_am_mc_dm).T

# Apply analysis operator T to obtain v_hat_prime 
gamma_km_mc_dm = np.einsum('ak, am -> km', F_ak_dm, b_am_mc_dm, dtype = float)

g_mc_dm = g_mc_dm[:K, :, :]
eta_qlm_mc_dm = np.einsum('qkl, km -> qlm', g_mc_dm, gamma_km_mc_dm, dtype = float)

c_mc_dm = c_mc_dm[:J, :, :]
v_hat_prime_mc_dm = np.einsum('qlm, plm -> pq', eta_qlm_mc_dm, c_mc_dm, dtype = float)

v_hat_prime_mc_dm = np.reshape(np.array(v_hat_prime_mc_dm), (J*K,1))

# %%
# Apply dual Gram operator G^+ to obtain v_hat 
# Using pushforward vF and original vector field v
# Both with Monte Carlo integration with weights
v_hat_mc_dm = np.matmul(G_dual_mc_dm, v_hat_prime_mc_dm)
v_hat_mc_dm = np.reshape(v_hat_mc_dm, (J, K))

# %%
# Apply pushforward map F_* of embedding F to v_hat to obtain approximated vector fields
# Using Monte Carlo integration with weights

h_ajl_mc_dm = np.einsum('ak, jkl -> ajl', F_ak_dm, g_mc_dm, dtype = float)

d_jlm_mc_dm = np.einsum('ij, ilm -> jlm', v_hat_mc_dm, c_mc_dm, dtype = float)

p_am_mc_dm = np.einsum('ajl, jlm -> am', h_ajl_mc_dm, d_jlm_mc_dm, dtype = float)


pushforward = p_am_mc_dm @ (eigvecs[:,0:I].T)

#True VF
mo.plot_vf(X[:,0:1000],v[:,0:1000])

#SEC approximated VF
mo.plot_vf(X[:,0:1000],pushforward[:,0:1000])



# %% 
# Apply our method:
# Reconstruction parameter:
t = .001
rank = np.min((I,J,K))
basis = eigvecs.T

# Construct component functions used to identify the components of the given
# vector field

x_1 = X[0,:]
x_2 = X[1,:]
x_3 = X[2,:]

#Truncate operator theoretic representation into square matrix
v = v_hat_mc_dm

v_inner = mo.op_to_inner(v)

# Multiplication operators:
m_x1 = mo.f_to_mult_op(x_1,basis,rank) 
m_x2 = mo.f_to_mult_op(x_2,basis,rank)
m_x3 = mo.f_to_mult_op(x_3,basis,rank)

v_x1 = v_inner(m_x1)
v_x2 = v_inner(m_x2)
v_x3 = v_inner(m_x3)


v_nc_reconstruct = np.zeros((3,num_pts))
for i in range(num_pts):
    v_nc_reconstruct[0,i] = mo.op_ker_eval(v_x1, X, i, basis, kernel=None)
    v_nc_reconstruct[1,i] = mo.op_ker_eval(v_x2, X, i, basis, kernel=None)
    v_nc_reconstruct[2,i] = mo.op_ker_eval(v_x3, X, i, basis, kernel=None)


# NONCOMMUTATIVE RECONSTRUCTION
mo.plot_vf(X[:,0:1000],v_nc_reconstruct[:,0:1000])
mo.plot_vf(X[:,0:1000],100*v_nc_reconstruct[:,0:1000])
