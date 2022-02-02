#!/usr/bin/env python
# coding: utf-8
import numpy as np

def construct_matrices(F, T, sigma):
    J, K = F.shape # F = observed fluxes, J = no. cameras, K = no. observations
    J1, K1, N = T.shape # T = common mode trends, N = no. trends (per camera)
    J2, K2 = sigma.shape # sigma = flux uncertainties
    assert J2 == J1 == J
    assert K2 == K1 == K
    M = J * N
    w = 1. / sigma**2
    Fw = F * w 
    Cinv = np.diag(1./w.sum(axis=0))
    D = np.zeros((K,M))
    E = np.zeros((M,M))
    for j in range(J):
        for n1 in range(N): 
            m1 = j * N + n1
            for k in range(K):
                D[k,m1] = T[j,k,n1] / sigma[j,k]**2
            for n2 in range(N):
                m2 = j * N + n2 
                E[m1, m2] = (T[j,:,n1] * T[j,:,n2] * w[j,:]).sum()
    y = np.zeros(M+K)
    y[:K] = Fw.sum(axis=0)
    for j in range(J):
        FwT = Fw[j,:][:,None] * T[j,:,:]
        y[K+j*N:K+(j+1)*N] = FwT.sum(axis=0)
    return Cinv, D, E, y

from scipy.linalg import inv
#from pyaneti import cholesky_inv_det, inverse,invert_multi_gp_matrix
def solve(F, T, sigma):
    J, K = F.shape # F = observed fluxes, J = no. cameras, K = no. observations
    J1, K1, N = T.shape # T = common mode trends, N = no. trends (per camera)
    J2, K2 = sigma.shape # sigma = flux uncertainties
    assert J2 == J1 == J
    assert K2 == K1 == K
    Cinv, D, E, y = construct_matrices(F, T, sigma)
    CinvD = np.dot(Cinv, D)
    DTCinvD = np.dot(D.T, CinvD)
    E_DTCinvD = E - DTCinvD
    E_DTCinvD_inv = inv(E_DTCinvD)
    bottomright = E_DTCinvD_inv
    topright = - np.dot(Cinv,np.dot(D,E_DTCinvD_inv))
    bottomleft = topright.T
    topleft = Cinv + np.dot(Cinv, np.dot(D, -bottomleft))
    Xinv = np.zeros((K+J*N,K+J*N))
    Xinv[:K,:K] = topleft
    Xinv[:K,K:] = topright
    Xinv[K:,:K] = bottomleft
    Xinv[K:,K:] = bottomright
    p = np.dot(Xinv,y)
    J, K, N = T.shape
    a = p[:K]
    w = p[K:].reshape((J,N))
    B = np.zeros((J,K))
    for j in range(J):
        for n in range(N):
            B[j,:] += w[j,n] * T[j,:,n]
    return a, w, B, (Cinv, D, E, y, CinvD, DTCinvD, E_DTCinvD, E_DTCinvD_inv, bottomright, topright, bottomleft, topleft, Xinv, p)

#Functions to correct the data using the CBVs

#Merit function definition -> http://mathworld.wolfram.com/MeritFunction.html
#Calculate the least squares between the raw flux and the basis vectors with a given set of parameters
#to be called by CBVCorrector
def merit_function(pars,flux,basis,n=2):
    squares = [0.0]*len(flux)
    for i in range(len(flux)):
        dummy = 0.0
        for j in range(len(basis)):
            dummy += pars[j]*basis[j][i]
        squares[i] = (abs(flux[i] - dummy))**n
    return np.sum(squares)

#Fucntion that corrects the flux "flux" given a set of CVBs "basis"
from scipy.optimize import fmin, minimize
def CBVCorrector(flux,basis):
        pars = [1.]*len(basis)
        #print("Correcting with {} basis vectors".format(len(basis)))
        pars = minimize(merit_function,pars,args=(flux,basis))
        new_flux = np.array(flux)
        #print('Coefficients = {}'.format(pars.x))
        #correct the raw flux with basis vectors
        for i in range(len(pars.x)):
            new_flux = new_flux - pars.x[i]*basis[i]
        return new_flux