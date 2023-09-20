#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 10:23:52 2022

@author: Ryan Vaughn
"""
# Debugger
import pdb


import numpy as np
import scipy.sparse
from scipy.linalg import eig
from scipy.sparse.linalg import eigs
import scipy.special
import scipy
from scipy.spatial.distance import pdist, squareform

from scipy.sparse.csgraph import laplacian

from itertools import product

import matplotlib.pyplot as plt
from matplotlib import cm,colors

import scipy.optimize
import scipy.interpolate


"""
This is a module containing functions for implementation of a non-commutative finite rank 
approximation scheme for vector fields on Riemannian manifolds.
"""


def check_function(f):
    """ 
    Almost all functions in this module take a numpy vector as an input. This
    vector represents the Fourier coefficients of a complex-valued function on
    the circle. In order to keep indexing consistent for the case of the circle,
    we require that such Fourier coefficients come in pairs and that they are 
    ordered by eigenvalue from least to greatest. This means that the middle 
    entry of each vector corresponds to the constant eigenfunction.
    
    The check_function function handles all of the error checking on the input
    vector to make sure that the data type is correct and that it follows the 
    appropriate indexing convention (must be of odd length.) The function outputs
    some parameters which are co
    
    
    Parameters
    ----------
    f : array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function. The length of the vector must be a specific
        value in order to keep track of indexing of the complex Fourier
        coefficients.
        
    
    Returns
    -------
    num_pairs : int
        The number of eigenpairs which will be used for the purposes of running
        the algorithm. Must be odd.
        
    num_eigs : int 
        The number of total eigenvalues. num_eigs := 2 * num_pairs + 1.

    valid : bool
        A bool indicating whether or not the input function has passed all
        testing requirements. Other functions will throw an error if valid == false.

    """
    valid = True
    f = f.ravel()
    num_eigs = f.shape[0]
    if f.shape[0] != num_eigs:
        raise Exception(
            "Vector f must be of odd length.(Circular eigenfunctions come in pairs)"
            )
        valid = False
        
    num_pairs = int((num_eigs-1)/2)
    if valid == False:
            print("Input vector failed formatting standards.")
    return num_pairs, num_eigs, valid

    
def f_to_mult_op_circle(f, num_pairs, sparse=True):
    
    """ 
    Construct a Toeplitz matrix (constant diagonal values) whose diagonals
    correspond to the entrees of the input vector f. The resulting matrix
    represents left multiplication by f. 
    
    For an example, let f and g be vectors of length num_eigs with complex
    entrees. These vectors are to be interpreted as collections of the complex
    Fourier coefficients of functions f and g respectively. 
    
    
    Parameters
    ----------
    f : array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function. The length of the vector must be a specific
        value in order to keep track of indexing of the complex Fourier
        coefficients.
        
    num_pairs : int
        Specifies the number of complex Fourier coefficients to be stored. For
        convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
        the number of total eigen functions used, including the constant
        function.
    
    Returns
    -------
    mult_op : array-like, shape=(num_eigs,)
        Matrix which represents left function multiplication by the function
        represented by the input vector.

    """

    
    "Error handling for appropriate size and shape of f."
    num_eigs = 2 * num_pairs + 1
    f = f.ravel()
    if f.ravel().shape[0] != num_eigs:
        raise Exception("Input function f is not of length num_eigs.")
    
    # Construct a list of all the off-diagonals of the Toeplitz matrix.
    eigs = np.arange(-num_pairs, num_pairs+1)
    diag_sizes = (num_eigs * np.ones(num_eigs) - np.abs(eigs)).astype(int)
    diags_list = [f[num_eigs-k-1]*np.ones(diag_sizes[num_eigs-k-1]) for k in range(num_eigs)]
   
    #Build the sparse Toeplitz matrix.
    mult_op =  scipy.sparse.diags(diags_list, eigs)
    
    #Return sparse or dense depending on the input.
    if sparse == False:
        return np.array(mult_op.todense())
    else:
        return mult_op


def mult_op_to_f_circle(m_f, sparse=True):
     
    """ 
    The inverse operation to f_to_mult_op_circle.
    
    
    Parameters
    ----------
    m_f : array-like, shape=(2*num_pairs+1,2*num_pairs+1)
        Matrix which represents left function multiplication by the function
        represented by the input vector.
    
    Returns
    ---------        
    f: A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function. The length of the vector must be a specific
        value in order to keep track of indexing of the complex Fourier
        coefficients.
    """
    
    (num_eigs,_)= m_f.shape
    num_eigs = int(num_eigs)
    num_pairs = int((num_eigs - 1) / 2)
    
    f = np.zeros(num_eigs)
    f[0:num_pairs + 1 ] = np.flip(m_f[0:num_pairs + 1,0])
    f[num_pairs:num_eigs + 1] = m_f[0,0:num_pairs + 1]
    return f


def f_to_toeplitz(f):
    
    """ 
    Construct a Toeplitz matrix (constant diagonal values) whose diagonals
    correspond to the entrees of the input vector f. The resulting matrix
    represents left multiplication by f (i.e. the left-regular representation.) 
    
    It's important to mention that composition of Toeplitz matrices naturally
    creates a loss of information for higher magnitude eigenfunctions. Given
    e^ai* e^bi = e^(a+b)i, if a+b is larger than the size of the matrix, the
    a+bth coefficient will be truncated.
    
    The f_to_toeplitz function is built to alleviate this slightly as compared
    to the f_to_mult_op_circle. Instead of mapping a function of size num_eigs
    to a num_eigs x num_eigs matrix, it sends a vector of length 
    2 * num_pairs + 1 to a matrix of size (num_pairs + 1,num_pairs + 1) thus 
    having less loss of information.
    
    The flipside of this is that one needs to ensure that the otput matrix
    is still of odd size if using other commands in this module to ensure proper
    bookkeeping of the eigenfunctions.
    
    
    Parameters
    ----------
    f : array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function. The length of the vector must be a specific
        value in order to keep track of indexing of the complex Fourier
        coefficients.

    Returns
    -------
    toeplitz_f : array-like, shape=(num_eigs,)
        Matrix which represents left function multiplication by the function
        represented by the input vector.

    """
    num_pairs,num_eigs,valid = check_function(f)
    
    if valid == True:
        toeplitz_f = np.zeros((num_pairs+1,num_pairs+1),dtype='complex')  
        
        for i in np.arange(-num_pairs,num_pairs+1):
            toeplitz_f = toeplitz_f + np.diag(f[i+num_pairs]*np.ones(num_pairs-abs(i)+1),i).astype(complex)
            
    else:
        print("f failed formatting check.")
        toeplitz_f = None
        
    return toeplitz_f

def truncate_function(f_max, max_num_pairs, num_pairs,):
    
    """
    Helper function which truncates a function f to a smaller number of Fourier
    coefficients and preserves the indexing convention (least to greatest 
    eigenvalues.)
    
    Parameters
    ----------
    f_max : array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function.
        
    max_num_pairs : int
        Specifies the number of pairs of complex Fourier coefficients of f_max.
        
    num_pairs : int
        Specifies the number of pairs of complex Fourier coefficients desired 
        in the output vector.
    
    """
    
    # Error handling for appropriate size and shape of f.
    f_max = f_max.ravel()
    if f_max.shape[0]:
        raise Exception("Input function f_max is not of length num_eigs.")
        
    f = f_max[(max_num_pairs - num_pairs):max_num_pairs + 1 + num_pairs]
    return f

def vfield(num_pairs,sparse = True):
    
    """
    Helper function that generates a matrix rep of the coordinate vector field 
    on the circle. This should be a matrix with ik on the diagonal entrees.
    
    Parameters
    ----------
    num_pairs : int
        Specifies the number of pairs of complex Fourier coefficients desired 
        in the output vector.
    """
    
    num_eigs = 2 * num_pairs + 1
    # Create array of diagonal entrees and fill them with ik for each eigenvalue k.
    diag_entrees = np.zeros(num_eigs,dtype=complex)
    diag_entrees = np.arange(-num_pairs,num_pairs+1)*1j
    
    # fill the diagonal matrix with the vector.
    vfield = scipy.sparse.spdiags(diag_entrees, 0 , num_eigs, num_eigs)
    
    # sparsity conditional.
    if sparse  == False:
        vfield = np.array(vfield.todense())
    return vfield

def inverse_fourier_transform_real(f,num_pairs):
    """ 
    Helper function for visualization which takes in a vector of fourier 
    coefficients and outputs a function which takes in data and outputs the real
    part of the spectrally truncated representation of f on the data.
    ----------
    f: array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigen functions used, including the constant
       function.
    """
    def pointwise_inv_fourier_f(x):
        
        coeffs = np.arange(-num_pairs, num_pairs+1)*1.j
        pointwise_real_part = 1 / (2*np.pi) * np.real(np.dot(f, np.exp(coeffs*x)))
        
        return pointwise_real_part

    #Vectorize the pointwise function so that it can be applied on data sets.  
    inv_fourier_f = np.vectorize(pointwise_inv_fourier_f)     
    
    return inv_fourier_f

def inverse_fourier_transform_imag(f,num_pairs):
    """ 
    Helper function for visualization which takes in a vector of fourier 
    functions and outputs a function which takes in data and outputs the 
    imaginary part of the spectrally truncated representation of f on the data.
    
    Parameters
    ----------
    f: array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigen functions used, including the constant
       function.
       
    Returns
    ----------
    
    """
    def pointwise_inv_fourier_f(x):
        
        coeffs = np.arange(-num_pairs, num_pairs+1)*1.j
        pointwise_imag_part = 1 / (2*np.pi) * np.imag(np.dot(f , np.exp(coeffs*x)))
        
        return pointwise_imag_part

    #Vectorize the pointwise function so that it can be applied on data sets  
    inv_fourier_f = np.vectorize(pointwise_inv_fourier_f)     
    
    return inv_fourier_f

def top_to_circulant(m_f,num_pairs):
    """ 
    A function which converts a toeplitz multiplication operator into a circulant
    matrix.
    ----------
    m_f: array-like, shape=(2*num_pairs+1, 2*num_pairs+1)
        A matrix which represents mulitiplication by f in the Fourier domain. 
        This most often will be the output of the function f_to_mult_op_circle.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigen functions used, including the constant
       function.
    """
    num_eigs = 2 * num_pairs + 1
    
    row = m_f[0] * 1
    col = m_f[:,0].T
    
    row[1:] = row[1:] + np.flip(col[1:])
    
    m_f_circ = np.zeros((num_eigs,num_eigs))*.0j
    
    for i in range(num_eigs):
        m_f_circ[i] = np.roll(row,i)
    
    return m_f_circ


def sec_prod(f,g,num_pairs):
    """ 
    Helper function which computes the product of f and g in the Fourier domain.
    This is what is done in the Spectral Exterior Calculus.
    ----------
    f,g: array-like, shape=(2*num_pairs+1,)
        Vectors whose entrees represent the complex Fourier coefficients of a
        complex valued function.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigen functions used, including the constant
       function.
    """
    
    num_eigs = 2 * num_pairs + 1
   
    #Very inefficient implemtentation of generating a list of indices (a,b)
    # such that |sum(a,b)|<= num_pairs
    indices = []
    iterate = range(-num_pairs,num_pairs+1)
    for i in product(iterate,iterate):
        sum_iterate = sum(i)
        if abs(sum_iterate) <= num_pairs:
            indices = indices + [i]
            
    #generate fg truncated up to num_eigs
    fg = np.zeros(num_eigs,dtype=complex)
    for i in indices:
        a  = i[0]
        b  = i[1]
        s = a+b
        
        fg[s + num_pairs] =  fg[s + num_pairs] + f[a+num_pairs] * g[b+num_pairs]
    
    return fg


def von_mises(mu,kappa,num_pairs):
    """ 
    Helper function which generates a Fourier representation of a von mises
    distribuition with mean mu and variance kappa.
    ----------
    mu: float
        The analogue of mean on the von mises distribution.
    kappa: float
        The analogue of variance on the von mises distribution.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigenfunctions used, including the constant
       function.
    """
    num_eigs = 2 * num_pairs + 1
    #The following loop computes the analytic Fourier coefficients of f.
    von_mises_dist = np.zeros(num_eigs,dtype='complex')
    
    for i in range(num_eigs):
        # reindex from entrees of i to the corresponding eigenvalue
        index = i - num_pairs
        von_mises_dist[i] = scipy.special.iv(np.abs(index),kappa)/scipy.special.iv(0, kappa)*np.exp(mu*1j)
    return von_mises_dist

def von_mises_reciprocal(mu,kappa,num_pairs):
    """ 
    Helper function which generates a Fourier representation of the reciprocal
    of the von mises distribution.
    
    ----------
    mu: float
        The analogue of mean on the von mises distribution.
    kappa: float
        The analogue of variance on the von mises distribution.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigenfunctions used, including the constant
       function.
    """
    num_eigs = 2 * num_pairs + 1
    #The following loop computes the analytic Fourier coefficients of f.
    von_mises_dist = np.zeros(num_eigs,dtype='complex')
    
    for i in range(num_eigs):
        # reindex from entrees of i to the corresponding eigenvalue
        index = i - num_pairs
        von_mises_dist[i] = scipy.special.iv(index,-1*kappa) *2*np.pi* scipy.special.iv(0,kappa)*np.exp(mu*1j)
    return von_mises_dist

    
def non_comm_inverse(f,num_pairs,eps):
    """ 
    Input function f, output is a function which takes in data and outputs the
    non-commutative spectral truncation of f on the data.
    ----------
    f: array-like, shape=(2*num_pairs+1,)
        Vector whose entrees represent the complex Fourier coefficients of a
        complex valued function.
        
   num_pairs : int
       Specifies the number of complex Fourier coefficients to be stored. For
       convenience, the value num_eigs is equal to 2 * num_pairs + 1 which is
       the number of total eigen functions used, including the constant
       function.
    """
    def pointwise_eigfuncts(x,num_pairs,eps):
        sqrt_eigvals = np.arange(-num_pairs,num_pairs + 1,dtype=complex)
        
        t1 = np.exp( -1 * sqrt_eigvals ** 2*eps) * np.exp(sqrt_eigvals*1.j*x)
        t2 = np.exp( -1 * sqrt_eigvals ** 2*eps) * np.exp(-sqrt_eigvals*1.j*x)
        pointwise_f_truncated = t1 @ f_to_mult_op_circle(f, num_pairs, sparse=False) @ t2 / (np.exp(-1* sqrt_eigvals) @ np.exp(-1* sqrt_eigvals))
        return pointwise_f_truncated
    
    non_comm = np.vectorize(pointwise_eigfuncts)
    
    return non_comm

def non_comm_truncation(f,eps):
    """ 
    Input a vector of Fourier coefficients of f and a tolerance variable, output
    is a function which takes in data and evaluates the non-commutative spectral
    truncation of f on the data. 
    
    Additionally, the function f must be of size which is 4*n+1 (i.e. 
    num_eigs - 1 is divisible by 4.) This is because the
    output is a matrix of size (num_pairs +1) x (num_pairs +1) so in order to 
    continue the indexing convention we must ensure that (num_pairs + 1) - 1 is
    divisible by two (otherwise check_function fails.)
    
    ----------
    f : array-like, shape = (2 * num_pairs + 1,)
        Vector whose entrees represent the complex Fourier coefficients of a
        complex valued function. 
       
    eps : float
        Bandwidth parameter of the kernel function.

      
    Returns
    ----------
    non_comm : function
        Function which takes in a real value and outputs the value of the
        non-commutatively spectrally truncated function on the input.
    bandlim_f : array-like, shape = (num_pairs + 1)
        Removes the tails of the original function f so that it may be applied
        to the mulitplication operator.
    """
    
    num_pairs,num_eigs,valid = check_function(f)
    
    num_bandlim_pairs = int(num_pairs/2) #rounds down
    bandlim_f = f[num_pairs-num_bandlim_pairs:2*num_pairs-num_bandlim_pairs+1]
    
    
    #TODO: I HAVE ADDED A TRANSPOSE HERE FOR TESTING PURPOSES. REMOVE IMMEDIATELY. TURNS OUT THE TRANSPOSE IS CORRECT, BUT RETURNS THE AVERAGE OF THE REAL VALUE OF F AS THE IMAGINARY PART. STRANGE. 
    
    if valid == True:
        def pointwise_nc_spectral_trunc(x,eps):
            sqrt_eigvals = np.arange(-num_bandlim_pairs,num_bandlim_pairs + 1,dtype=complex)
            
            t1 = np.exp(-sqrt_eigvals ** 2 * eps) * np.exp(sqrt_eigvals * 1.j * x)
            t2 = np.exp(-sqrt_eigvals ** 2 * eps) * np.exp(-sqrt_eigvals * 1.j * x)
            pointwise_f_truncated = t1 @ f_to_toeplitz(f).T @ t2 / np.sum(np.exp(-2* sqrt_eigvals ** 2 * eps)) /(2*np.pi)
            return pointwise_f_truncated
        
        non_comm = np.vectorize(pointwise_nc_spectral_trunc)
    else:
        non_comm = None
    
    return non_comm, bandlim_f
    
    
def data_inverse_fourier(f,eigenfunctions):
    """ 
    Input a function f in data space, together with an array whose columns are 
    empirically estimated Laplacian eigenfunctions on data. Output is the function
    f represented in "Fourier space." However, note that this can be computed for
    any manifold M, not just the circle.
    
    ----------
    f : array-like, shape = (num_pts)
        Vector whose entrees represent the values of a function on a given
        data set. The number of entrees thus correspond to the number of data
        points
       
    eigenfunctions : array-like, shape = (num_pts, num_eigs)
        Matrix whose columns correspond to Laplacian eigenfunctions computed on
        a data set.
      
    Returns
    ----------
     f_hat : array-like, shape = (num_eigs,)
         Vector whose entrees represent the Fourier coefficients of f arranged
         in an order corresponding to those generated by the Diffusion maps
         algorithm.
    """
    (num_pts, num_eigs) = np.shape(eigenfunctions)
    
    f_hat = (eigenfunctions @ f) / num_pts
    
    return f_hat

"""
This is a quick implementation of the Diffusion Maps Algorithm
"""

def gen_dm_laplacian(X,eps,num_eigs): 
    """ 
    Given input sample data X and parameter eps, generate an rbf kernel matrix
    affinity. Perform diffusion maps and an eigendecomposition to return Laplacian Eigenfunctions.
    
    ----------
    X : array-like, shape = (dim, num_pts)
       Input sample data.

    eps : float
       Scaling Parameter
      
    num_eigs : int
       Number of eigenvalues to compute
       
    Returns
    ----------
     
    """
    (dim, num_pts) = X.shape
    
    euclidean_dist_sq = squareform(pdist(X.T)**2) 
    aff = np.exp(-(euclidean_dist_sq) / (eps**2))
    # For laplacian Eigenvalues, we use alph=1, but leaving this in in case we
    # someday are interested in taking weighted eigenvalues
    alph = 1
    
   
    
    #left normalize
    d = np.diag((np.ones(num_pts) @ aff)**-alph)
    aff = d @ aff
    
    #right normalize
    d = np.diag((np.ones(num_pts) @ aff)**-alph)
    aff =  aff @ d 
    
    #Symmetric Normalize
    d_half = np.diag((np.ones(num_pts) @ d)**(-1/2))
    L = d_half @ aff @ d_half

    eigvals,eigvecs = eigs(L,num_eigs,which ='LM')
    
    # For now, lets use the full eigendecomposition
    # eigvals,eigvecs = eig(L)
    
    #To get the right eigenvectors, we have to renormalize
    eigvecs = d_half @ eigvecs 
 
    #Sort the Eigenvalues and Eigenvectors
    inds = np.argsort(eigvals)
    inds = inds[::-1]
    eigvals = eigvals[inds]
    eigvecs = eigvecs[:,inds]
    
    
    return L,eigvals,eigvecs,aff

def gen_dm_laplacian_updated(X,eps,num_eigs): 
    """ 
    Given input sample data X and parameter eps, generate an rbf kernel matrix
    affinity. Perform diffusion maps and an eigendecomposition to return Laplacian Eigenfunctions.
    
    ----------
    X : array-like, shape = (dim, num_pts)
       Input sample data.

    eps : float
       Scaling Parameter
      
    num_eigs : int
       Number of eigenvalues to compute
       
    Returns
    ----------
     
    """
    (dim, num_pts) = X.shape
    
    euclidean_dist_sq = squareform(pdist(X.T))**2
    aff = np.exp(-euclidean_dist_sq / (eps**2))
    # For laplacian Eigenvalues, we use alph=1, but leaving this in in case we
    # someday are interested in taking weighted eigenvalues
    alph = 1
    
    d_negative_alpha = np.diag(np.sum(aff,axis=1) ** -alph)
                
    # normalize
    k_hat = d_negative_alpha @ aff @ d_negative_alpha
    
    d_hat_diag = np.sum(k_hat,axis=1)
    
    '''
    There is an intermediary step to make k_hat into a Markov matrix 
    P = (d_hat **-1) @ k_hat but we instead construct a similar matrix which has 
    the same eigenvectors.
    
    #Symmetric Normalize
    d_half = np.diag((np.ones(num_pts) @ d)**(-1/2))
    L = d_half @ aff @ d_half
    '''
    
    d_hat_negative_half = np.diag(d_hat_diag ** (-1/2))
    S = d_hat_negative_half @ k_hat @ d_hat_negative_half
    
    eigvals,eigvecs = scipy.linalg.eigh(S)

    #Renormalize eigenvalues so they correspond to Laplacian eigenvalues then sort
    eigvals = - np.log(eigvals) / (eps ** 2)

    
    # Eigenvectors of S seem to give back a more accurate reconstruction
    # eigvecs = d_hat_negative_half @ eigvecs
    
    #normalized with L^2 norm 1
    eigvecs_magnitudes = np.sqrt(np.sum(eigvecs ** 2,axis=0)/num_pts)
    eigvecs_normalized = eigvecs/eigvecs_magnitudes
    
    #Sort in ascending order
    eigvals = eigvals[::-1]
    eigvecs_normalized =np.flip(eigvecs_normalized,axis=1)
    
    #Return only num_eigs
    eigvals = eigvals[0:num_eigs]
    eigvecs_normalized = eigvecs_normalized[:,0:num_eigs]
    
    return S,eigvals,eigvecs_normalized,aff


def gen_le_laplacian(X,eps,num_eigs): 
    """ 
    For testing purposes, we use the built in graph laplacian for uniform data.
    ----------
    X : array-like, shape = (dim, num_pts)
       Input sample data.

    eps : float
       Scaling Parameter
      
    num_eigs : int
       Number of eigenvalues to compute
       
    Returns
    ----------
     
    """
    (dim, num_pts) = X.shape
    
    euclidean_dist_sq = squareform(pdist(X.T)**2) 
    aff = np.exp(-(euclidean_dist_sq) / (eps**2))
    # For laplacian Eigenvalues, we use alph=1, but leaving this in in case we
    # someday are interested in taking weighted eigenvalues
    
    L_le = laplacian(aff)

    # FIX THIS
    #eigvals,eigvecs = eigs(aff,num_eigs,which ='LM')
    
    # For now, lets use the full eigendecomposition
    eigvals,eigvecs = eig(L_le)
    
    #Sort the Eigenvalues and Eigenvectors
    inds = np.argsort(eigvals)
    eigvals = eigvals[inds]
    eigvecs = eigvecs[:,inds]
    
    return L_le,eigvals,eigvecs

def dd_fourier_transform(eigvecs):
    """ 
    Helper function which takes in a basis function generated through diffusion
    maps on data.
    ----------
    eigvecs: array-like, shape=(num_pts, num_eigs)
        An array of vectors whose entrees represent the real values of a basis 
        function on finitely sampled data.
    Returns
    ----------
     f_hat : array-like, shape = (num_eigs)
         An affinity matrix generated by the rbf kernel.
    
    """
    (num_pts, num_eigs) = eigvecs.shape
    def dd_f_to_fourier(f):
        
        f_hat = np.zeros(num_eigs)
        
        for i in range(num_eigs):
            f_hat[i] = (1/num_pts)(eigvecs[:,i] @ f)
        
        return f_hat
    
    def dd_fourier_to_f(f_hat):
        
        num_entrees = f_hat.size
        f_hat = np.reshape(f_hat,(num_entrees,1))
        f = np.multiply(eigvecs,f_hat)
        f = np.sum(f,axis = 0)
        return f
    
    return dd_f_to_fourier,dd_fourier_to_f


def dd_f_to_multop(f,eigvals,eigvecs):
    
    """  
    
    Parameters
    ----------
    f : array-like, shape=(2*num_pairs+1,)
        A vector whose entrees represent the complex Fourier coefficients of a
        complex valued function. The length of the vector must be a specific
        value in order to keep track of indexing of the complex Fourier
        coefficients.

    Returns
    -------
    toeplitz_f : array-like, shape=(num_eigs,)
        Matrix which represents left function multiplication by the function
        represented by the input vector.

    """
    num_pairs,num_eigs,valid = check_function(f)
    
    if valid == True:
        toeplitz_f = np.zeros((num_pairs+1,num_pairs+1),dtype='complex')  
        
        for i in np.arange(-num_pairs,num_pairs+1):
            toeplitz_f = toeplitz_f + np.diag(f[i+num_pairs]*np.ones(num_pairs-abs(i)+1),i).astype(complex)
            
    else:
        print("f failed formatting check.")
        toeplitz_f = None
        
    return toeplitz_f

def dd_nc_truncation(f,eigvals,eigvecs,t):
    
    """  
    Function which takes in a function f on data points together with a matrix
    of approximate eigenfunctions on the data points. The output is a function
    on data which represents an approximation of f using our noncommutative method.
    
    
    Parameters
    ----------
    f : array-like, shape = num_pts,
        A vector whose entrees represent the values of  a function on input data.
    eigvals, eigvecs : array-like, shape = num_eigs,dim
        A collection of eigenvalues and eigenvectors generated from
    diffusion maps or another method
    t : float
        Scaling parameter which is independent of the data driven parameter but also scales tolerance of approximation.
    Returns
    -------
    f_nc : array-like, shape = num_pts,

    """
    num_pts = f.shape[0]
    
    #Step 1: Represent f as a multiplication operator
    # ---------------------------------------------------
    # Construct a num_eigs x num_eigs bilinear form whose entrees are int_m f * phi_j * phi_i dV 
    f_times_eigs = 1 / num_pts * (f * eigvecs.T).T
    
    L2_prod_f_f_times_eig = (eigvecs.T @ f_times_eigs) / num_pts
    # ---------------------------------------------------
    
    # Compute the spectrally truncated heat kernel.
    
    exps = np.exp(-1 * eigvals * t)
    l_vec = exps * eigvecs
    r_vec = exps * eigvecs
    
    #Construct a full array of entrees for which we will extract the trace so 
    # that we compute the trace of the kernel section together with f (i.e. approximate function evaluation)
    bilin_form = l_vec @ L2_prod_f_f_times_eig @ r_vec.T
    

    # Need to normalize by the L^2 norm of the truncated kernel to ensure that the result is a state.   
    normalizers = (eigvecs ** 2) @ (exps ** 2) / num_pts
    
    #The trace corresponds to the diagonal of the bilinear form.
    output = np.diag(bilin_form) / normalizers
    
    return output 

def spherical_grad(X,func,eps):
    dim,num_pts= X.shape
    gradient = np.zeros(X.shape)
    for i in range(num_pts):
        point = X[:,i]
        jacobian = scipy.optimize.approx_fprime(point,func,eps)
        ex_deriv = (jacobian @ point) * point
        cov_deriv = jacobian - ex_deriv
        gradient[:,i] = cov_deriv
        
    return gradient
# %% Plotting:
def inv_comp_ster_proj(X):
    ''' perform inverse stereographic projection on data and shrink to bounded interval
    using inverse tangent '''
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    
    x_p = 2 * np.pi * np.arctan(x/(1-z))
    y_p = 2 * np.pi * np.arctan(y/(1-z))
    return x_p,y_p

def plot_f_2d(x,y,f,title='',cm='seismic',marker_size=.1):
    #Plot the data
    ax = plt.axes()
    
    ax.scatter(x, y, c = np.real(f),cmap=cm,s=marker_size)
    plt.title(title)
    plt.show()    

def nc_truncation(X,eigvals,eigvecs,num_eigs,t):    
    (dim,num_pts) =X.shape
    weights = np.exp(-eigvals*t)
    h = np.zeros((num_pts, num_pts))
    for i in range(num_pts):
        for j in range(num_pts):
            for k in range(num_eigs):
                h[i,j] = weights[k]*eigvecs[i,k]*eigvecs[j,k]
                
    return h

def plot_eig(X,eigvecs,eig_num,cm='seismic',marker_size=.1):
    #Plot the data
    fig = plt.figure()
    ax = plt.axes(projection ='3d')
    
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    
    eig = eigvecs[:,eig_num]
    ax.scatter(x, y, z, c = eig, cmap=cm,s=marker_size)
    plt.show()

def plot_f(X,f,title,cm='seismic',marker_size=.1):
    #Plot the data
    ax = plt.axes(projection ='3d')
    
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    
    ax.scatter(x, y, z, c = np.real(f),cmap=cm,s=marker_size)
    plt.title(title)
    plt.show()
    
def plot_multiple_f(X,f,v_min,v_max,cm='seismic',marker_size=.1):
    #Plot the data
    ax = plt.axes(projection ='3d')
    
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    
    ax.scatter(x, y, z, c = np.real(f),vmin =v_min ,vmax =v_max,cmap=cm,s=marker_size)
    plt.show()
    
def spherical_harmonic(X,m,n):
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    
    theta = np.arccos(z) 
    phi =np.sign(x) * x / np.sqrt(x**2 +y**2)
    
    sph_harmonic = scipy.special.sph_harm(m,n,theta,phi)     
    re_harmonic = np.real(sph_harmonic)
    im_harmonic = np.real(sph_harmonic)
    
    return re_harmonic,im_harmonic

def plot_comparison_graphic(X,num_eigs,t,eps,f,f_nc,v_min,v_max,cm_string='seismic',marker_size=.1):

    #Plot the data
    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax = fig.add_subplot(2,1,1,projection='3d')
    ax.set_facecolor('green')
    plt.title('Actual Function')
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
   
    ax.scatter(x, y, z, c = np.real(f),vmin =v_min,vmax=v_max,cmap=cm_string,s=marker_size)
    
    
    ax = fig.add_subplot(2,1,2,projection='3d')
    ax.set_facecolor('green')
    ax.scatter(x, y, z, c = np.real(f_nc),vmin =v_min,vmax=v_max,cmap=cm_string,s=marker_size)
    plt.title('NC Estimate (L = ' + str(num_eigs) + ', t = ' + str(t) + ', e = ' + str(eps) + ' )')
    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
    
   
    # define color map
    cmap = cm.get_cmap(cm_string)

    # need to normalize because color maps are defined in [0, 1]
    norm = colors.Normalize(v_min, v_max)
    
    # plot colorbar
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    
    fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap),ax = cbar_ax, cax = cbar_ax)
    plt.show()
    
def plot_vf(X,v):
    plt.clf()
    #Plot the data
    ax = plt.axes(projection ='3d')
    
    x = X[0,:]
    y = X[1,:]
    z = X[2,:]
    
    a = v[0,:]
    b = v[1,:]
    c = v[2,:]
    ax.quiver(x,y,z,a,b,c,length =0.1)
    plt.show()

def plot_2d_vf(X,v):
    plt.clf()
    #Plot the data
    ax = plt.axes()
    x = X[0,:]
    y = X[1,:]
    
    a = v[0,:]
    b = v[1,:]
    
    lengths_sq = np.sum(v * v,axis = 0)
    max_length = np.max(np.sqrt(lengths_sq))
    
    ax.quiver(x,y, a,b,scale= 5 * max_length)
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def eig_to_func(X,eigvecs,eig_num):
    def eigfunc(x):
        dim,num_pts = X.shape
        eig = eigvecs[:,eig_num]
        
        c1 = (X[0,:] == x[0])
        c2 = (X[1,:] == x[1])
        c3 = (X[2,:] == x[2])         
        where = c1 * c2* c3
        return eig[where][0]
    return eigfunc



def grad(X,func,eps):
    dim,num_pts= X.shape
    gradient = np.zeros(X.shape)
    for i in range(num_pts):
        point = X[:,i]
        jacobian = scipy.optimize.approx_fprime(point,func,eps)
        ex_deriv = (jacobian @ point) * point
        cov_deriv = jacobian - ex_deriv
        gradient[:,i] = cov_deriv
        
    return gradient

def dd_fourier_trunc(X,f,eigvecs,k):
    dim, num_pts = X.shape
    coeffs = (f @ eigvecs)/num_pts
    weighted_eigs = coeffs * eigvecs
    
    return weighted_eigs

# %% June 2023 Clean Up
def f_to_mult_op(f,basis,rank):
    
    '''
    Description
    -----------
    
    A function which transforms a vector representing values of a function f 
    on point cloud data into a finite-rank approximation of the 
    multiplication operator by f with respect to the input L^2 basis whose
    columns are approximate L^2 orthonormal basis functions. (Here we assume 
    that the L^2 inner product is weighted by the underlying sampling density.)
    
    Parameters
    ----------
    
    f : array-like, shape = num_pts,
        A vector whose entrees represent the values of  a function on input
        data.
    
    basis : array-like, shape = num_basis, num_pts
        An array whose columns are data-driven approximation to the first
        entrees L2 orthonormal basis of functions on the data set.
    
    rank : int, 
        The chosen rank of the output multiplication operator. Must be less 
        than the number of columns of eigvecs.
    
    -------
    f_op : array-like, shape = (rank, rank)
        An operator theoretic representation of the input function with 
        respect to the L^2 basis of functions approximated by the columns of
        basis. Such an operator takes in a vector of length rank whose entrees
        correspond to the L^2 coefficients of a function g with respect to the
        input basis.
    
    '''
    (num_basis, num_pts) = basis.shape
    
    # Check that the output rank is not greater than the number of input basis functions
    if num_basis < rank:
        print("The rank of multiplication operator exceeds the number of input basis functions.")
    
    # Construct the multiplication operator corresponding to f
    m_f = np.zeros((rank,rank))
    for j in range(rank):
        for k in range(rank):
                m_f[j,k] = (1 / num_pts) * np.sum(f * basis[j,:] * basis[k,:])
    return m_f
        

def op_to_inner(op):
    
    '''
    Description
    -----------
    A helper function which maps finite-rank approximation of an operator to 
    an inner derivation. 
    
    Parameters
    ----------
    
    op : array-like, shape = rank, rank
        A finite rank operator 
    -------
    inner_op : function
        A function which takes in a square matrix of size rank and outputs the 
        commutator of op with the input matrix.
    '''
    
    def inner_op(A):
        return op @ A - A @ op
    
    return inner_op

def ker_to_sec(X, point, kernel=None):
    
    '''
    Description
    -----------
    Helper function which takes a kernel function together with sampled data X
    and specified point in X and outputs the kernel section corresponding to
    k(point,.) evaluated on the data set.
    
    Parameters
    ----------
    op : array-like, shape = (rank, rank)
        A matrix which represents a finite-rank approximation to a linear 
        operator acting on L^2 functions.
        
    kernel : function, kernel(x, y)
        A function which takes in two data points and outputs a positive
        real value. The kernel function is assumed to have exponential
        decay. That is: k(eps,x,y) < A * np.exp(-B*((x - y) @ (x - y))
        /(eps**2)) for some positive A, B.
    
    X : array-like, size = (dim, num_pts)
        An array whose columns correspond to point cloud data. Assumed to be 
        the underlying data set on which the approximate L2 basis functions 
        are computed.
    
    point : array-like, size = dim,
        A specific point for which the value of the operator is to be 
        reconstructed (assumed to be a point in the underlying data set X.)
        
    -------
    k_section : array-like, size = num_pts,
        Array whose entrees correspond to k(point, x_i) for each x_i in X.
    '''

    (num_basis, num_pts) = X.shape 
    
    # The default kernel is an RBF kernel with pre-selected bandwidth parameter.    
    if kernel == None:
        eps = 0.1
        gaussian_kernel = lambda x,y : np.exp(-((x-y).T @ (x-y))/(eps**2))
        kernel = gaussian_kernel
    
    k_section = np.array([kernel(point,X[:,i]) for i in range(num_pts)])
    
    return k_section

def f_to_trunc(f,basis,rank):
    
    '''
    Description
    -----------
    Helper function which takes a function f represented as a vector of
    length num_pts and an array whose colums are approximate L^2 orthonormal
    basis of functions all with respect to some underlying data set X, outputs
    a vector of the first rank (input variable) coefficients of f with respect
    to the input basis.
    
    Parameters
    ----------
    
    f : array-like, shape = num_pts,
        A vector whose entrees represent the values of  a function on input
        data.
    
    basis : array-like, shape = num_basis, num_pts
        An array whose columns are data-driven approximation to the first
        entrees L2 orthonormal basis of functions on the data set.
    
    rank : int, 
        The chosen rank of the output multiplication operator. Must be less 
        than the number of columns of eigvecs.
    
    -------
    f_hat : array-like, shape = rank
        The first rank coefficients of f with respect to the input orthonormal basis.
    '''
    
    (num_basis, num_pts) = basis.shape
    
    # Check that the output rank is not greater than the number of input basis functions.
    if num_basis < rank:
        print("The rank of multiplication operator exceeds the number of input basis functions.")
        return None
    
    # Construct a vector of the first rank fourier coefficients of f.
    f_hat = (1 / num_pts) * basis[0:rank,:] @ f
    return f_hat

def trunc_to_proj(f_hat,basis):
    
    '''
    Description
    -----------
    Helper function which takes a list of generalized fourier coefficients with
    respect to the provided finite approximate basis. Output is an operator
    which takes in a list of coefficients of the same size as f and output is
    the projection of the input onto f.
    
    Parameters
    ----------
    
    f_hat : array-like, shape = rank,
        A vector whose entrees represent the values of  a function on input
        data.
    
    basis : array-like, shape = num_basis, num_pts
        An array whose columns are data-driven approximation to the first
        entrees L2 orthonormal basis of functions on the data set.
    
    rank : int, 
        The chosen rank of the output multiplication operator. Must be less 
        than the number of columns of eigvecs.
    
    -------
    proj_f : array-like, shape = (rank,rank)
        The projection operator onto f_hat, which lives in the finite rank
        subspace spanned by the first rank basis functions.
    '''
    
    rank = f_hat.shape[0]
    
    proj_f = np.diag(f_hat)
    
    return proj_f
    
def op_ker_eval(op, X, point, basis, kernel=None):
    
    '''
    Description
    -----------
    A helper function combining the functions ker_to_sec and f_to_trunc. Takes in a kernel function and
    outputs a state corresponding to the input function f.
    
    Parameters
    ----------
    op : array-like, shape = (rank, rank)
        A matrix which represents a finite-rank approximation to a linear operator acting on L^2 functions.
        
    kernel : function, kernel(x, y)
        A function which takes in two data points and outputs a positive real value.
    
    X : array-like, size = (dim, num_pts)
        An array whose columns correspond to point cloud data. Assumed to be the underlying data set on which the approximate 
        L2 basis functions are computed.
    
    point : array-like, size = dim,
        A specific point for which the value of the operator is to be reconstructed (assumed to be a point in the underlying
        data set X.)
    
    -------
    op_x : A generalized pointwise evaluation at the given point.
    
    '''
    
    (rank,rank_check) = op.shape
    if rank_check != rank:
        print("The input operator is not square.")
        return None
    
    k_sec = ker_to_sec(X,point,kernel)
    sq_k_sec = np.sqrt(k_sec)
    sq_k_sec_hat = f_to_trunc(sq_k_sec,basis,rank)
    
    # The vector must be normalized so that it is in fact a state.
    sq_k_sec_hat_mag = np.sqrt(sq_k_sec_hat @ sq_k_sec_hat)
    
    op_x = sq_k_sec_hat.T @ op @ sq_k_sec_hat
    
    return op_x