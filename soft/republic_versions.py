#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv, cho_solve, cho_factor, svd

# Common variable names:
# - F: 2-D array of input fluxes for one star, multiple cameras. 
#      Shape (J,K) where:
#      - J = no. cameras, 
#      - K = no. observations
# - T: 3-D array of pre-computed trends (CBVs) for each camera
#      Shape (J, K, N) where:
#      - J, K as before
#      - N = no. trends (assumed to be the same for all cameras)
# - sigma: 2-D array of flux uncertainties
#      Shape (J, K)
# - t: 1-D array of times
#      Shape (K,)

### PDC-LS ###

def pdc_ls(F, T, sigma):
    # linear least squares fit of CBVs to LC in each quarter
    # Inputs: (F, T, Sigma)
    # returns: (Fc, Fc_av, F_av) where
    # - Fc = corrected flux, one per camera, shape (J, K)
    # - Fc_av = corrected flux averaged across all cameras, using inverse variance weighting, shape (K,)
    # - F_av = raw flux averaged across all cameras (for comparison), shape (K,)
    J, K, N = T.shape 
    w = 1. / sigma**2
    Fw = F * w 
    Fc = np.copy(F)
    for j in range(J): # loop over cameras
        X = np.zeros((N,N))
        for n1 in range(N): 
            for n2 in range(N):
                X[n1, n2] = (T[j,:,n1] * T[j,:,n2] * w[j,:]).sum()
        FwT = Fw[j,:][:,None] * T[j,:,:]
        y = FwT.sum(axis=0)
        p = solve(X,y)
        cor = np.dot(T[j,:,:],p)
        Fc[j,:] -= cor
    # average across cameras
    w_av = np.mean(w, axis=1)
    F_av = np.average(F, axis=0, weights = w_av)
    Fc_av = np.average(Fc, axis=0, weights = w_av)
    return Fc, Fc_av, F_av

def mk_basis(t, M, basis = 'sin', include_bias = True, doplot=True, normalise = True):
    # construct basis from specified family of functions
    # Inputs: (t, M, basis, include_bias, doplot, normalise) where
    # - M = number of terms in basis (actual number of basis functions can differ 
    #       depending on basis type and whether a bias term is included)
    # - basis = 'fourier' (default), 'gauss', 'poly', 'tanh'
    # - include_bias = True (default) / False
    # - doplot = True (default) / False
    # - normalise = True (default) / False 
    # Outputs: (A) where
    # - A = 2-D matrix of basis functions evaluated at times t, shape (L,K) where L = M, 2*M, M+1 or 2*M+1
    K = len(t)
    tmin, tmax = np.nanmin(t), np.nanmax(t)
    T = tmax - tmin
    dt = np.nanmedian(t[1:]-t[:-1])
    if basis == 'const':
        A = np.ones((1, K))
        normalise = False
        include_bias = False
    elif basis == 'eye':
        A = np.eye(K)
        normalise = False
        include_bias = False
    elif basis == 'sin':
        A = np.zeros((M, K))
        for m in range(M):
            p = T / (m+1)
            ph = np.random.uniform(0,2*np.pi)
            A[m,:] = np.sin(2 * np.pi * t / p + ph)
    elif basis == 'fourier':
        A = np.zeros((M*2,K))
        for m in range(M):
            arg = (m+1) * (2*np.pi) * (t / T)
            A[2*m,:] = np.cos(arg)
            A[2*m+1,:] = np.sin(arg)
    elif basis == 'gauss':
        sig = T/(M-5)
        mus = np.linspace(tmin-sig,tmax+sig,M)
        A = np.zeros((M,K))
        for m in range(M):
            arg = (t-mus[m])/sig
            A[m,:] = np.exp(-arg**2)
    elif basis == 'poly':
        A = np.zeros((M,K))
        arg = (t-tmin)/T-0.5
        for m in range(M):
            A[m,:] = arg**(m+1)
    elif basis == 'tanh':
        sig = T/(M-3)
        mus = np.linspace(tmin-sig,tmax+sig,M)
        A = np.zeros((M,K))
        for m in range(M):
            arg = (t-mus[m])/sig
            A[m,:] = np.tanh(arg)
    else:
        print('ERROR: Basis function type {} not recognised'.format(basis))
        return
    if normalise:
        for m in range(M):
            tmp = A[m,:]
            tmp -= tmp.mean()
            A[m,:] = tmp / tmp.std()
    if include_bias:
        A = np.concatenate([np.ones_like(t).reshape((1,K)),A])
    if doplot:
        fig,ax = plt.subplots(2,1,sharex=True)
        ax[0].plot(t, A.T,lw=0.5)
        ax[0].set_title('{} basis'.format(basis))
        ax[0].set_ylabel('basis')
        for i in range(10):
            m = A.shape[0]
            w = np.random.uniform(-1/m,1/m,m)
            ax[1].plot(t, np.dot(w,A))
        ax[1].set_ylabel('examples')
        ax[1].set_xlabel('input var.')
        ax[1].set_xlim(t.min(),t.max())
        plt.show()
    return A

def mk_segmented_basis(t, s, K, basis = 'poly', include_bias = True, doplot=True):
    N = len(t)
    su = np.unique(s)
    Ns = len(su)
    for i,ss in enumerate(su):
        As = mk_basis(t, K, basis = basis, include_bias = include_bias, doplot = False)
        l = s != ss
        As[:,l] = 0
        if i == 0:
            A = np.copy(As)
        else:
            A = np.concatenate([A,As])
    if doplot:
        fig,ax = plt.subplots(2,1,sharex=True)
        ax[0].set_title('segmented {} basis'.format(basis))
        ax[0].set_ylabel('basis')
        ax[0].plot(t, A.T,lw=0.5)
        ax[1].set_ylabel('examples')
        ax[1].set_xlabel('input var.')
        for i in range(10):
            k = A.shape[0]
            w = np.random.uniform(-1/k,1/k,k)
            ax[1].plot(t, np.dot(w,A))
        ax[1].set_xlim(t.min(),t.max())
        plt.show()
    return A

def republic(t, F, sigma, T = None, s = None, \
             star_basis_type = 'eye', L_in = 6, \
             sys_basis_type = 'poly', N_in = 3, \
             lambda_reg = 0.0):
    # for republic 1, supply T and let star_basis_type = 'eye'
    # for republic 2, don't supply T and set star_basis_type to (e.g.) 'sin'
    # for republic 3, supply T and set star_basis_type to (e.g.) 'sin'
    # for PDC-LS, supply T and set star_basis_type to 'const'

    J,K = F.shape
    # If star_basis_type = 'eye' (default), A is a (K,K) identity matrix (stellar flux is free at time-step)
    # If star_basis_type = 'const' A is a (1,K) matrix set to unity (stellar flux assumed constant)

    if star_basis_type == 'eye':
    # Republic 1: user-supplied CBVs + free stellar signal
        if T is None:
            print('Error: must supply T if star_basis_type is "eye"')
            return
        # top left part of design matrix is large but diagonal, so compute its inverse directly
        w = 1./sigma**2
        Fw = F * w
        Cinv = np.diag(1./w.sum(axis=0))
        N = T.shape[-1]
        M = J * N
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
        y = np.zeros(K + M)
        y[:K] = Fw.sum(axis=0)
        for j in range(J):
            FwT = Fw[j,:][:,None] * T[j,:,:]
            y[K+j*N:K+(j+1)*N] = FwT.sum(axis=0)
        # solve for coefficients
        p = np.dot(Xinv,y)
    else:            
        # construct A matrix
        A = mk_basis(t, L_in, basis = star_basis_type, include_bias = False, doplot=False)
        L = A.shape[0]
        if T is None: 
            # Republic 2: use functional basis for both star and systematics
            if s is None:
                s = np.zeros(K).astype(int)
            B = mk_segmented_basis(t, s, N_in, basis = sys_basis_type, include_bias = True, doplot=False)
            N = B.shape[0]
            P = L + J * N
            X = np.zeros((P,P))
            y = np.zeros(P)
            X[:L,:L] = ((A[:,None,:] * A[None,:,:])[:,:,None,:] / (sigma[None,None,:,:])**2).sum(axis=-1).sum(axis=-1)
            y[:L] = (A[:,None,:] * (F / sigma**2)[None,:,:]).sum(axis=-1).sum(axis=-1)
            for j in range(J):
                D = ((A[:,None,:] * B[None,:,:]) / sigma[j,:]**2).sum(axis=-1)
                X[:L,L+j*N:L+(j+1)*N] = D
                X[L+j*N:L+(j+1)*N,:L] = D.T
                E = ((B[:,None,:] * B[None,:,:]) / sigma[j,:]**2).sum(axis=-1)
                X[L+j*N:L+(j+1)*N,L+j*N:L+(j+1)*N] = E
                y[L+j*N:L+(j+1)*N] =  (B * (F / sigma**2)[None,j,:]).sum(axis=-1)
        else:
            # Republic 3: use functional basis for star and user-supplied CBVs for systematics
            N = T.shape[-1]
            P = L + J * N
            X = np.zeros((P,P))
            y = np.zeros(P)
            X[:L,:L] = ((A[:,None,:] * A[None,:,:])[:,:,None,:] / (sigma[None,None,:,:])**2).sum(axis=-1).sum(axis=-1)
            y[:L] = (A[:,None,:] * (F / sigma**2)[None,:,:]).sum(axis=-1).sum(axis=-1)
            for j in range(J):
                B = T[j,:,:].T
                D = ((A[:,None,:] * B[None,:,:]) / sigma[j,:]**2).sum(axis=-1)
                X[:L,L+j*N:L+(j+1)*N] = D
                X[L+j*N:L+(j+1)*N,:L] = D.T
                E = ((B[:,None,:] * B[None,:,:]) / sigma[j,:]**2).sum(axis=-1)
                X[L+j*N:L+(j+1)*N,L+j*N:L+(j+1)*N] = E
                y[L+j*N:L+(j+1)*N] =  (B * (F / sigma**2)[None,j,:]).sum(axis=-1)
        # solve for coefficients using SVD
        X += lambda_reg * np.eye(P) # L2 regularisation, may need tuning
#        p = cho_solve(cho_factor(X), y)
        U, w, V_T = svd(X)
        z = 1. / w
        w_max = abs(w).max()
        thresh = 1e-12 * w_max
        small = w <= thresh
        if small.any():
            print('Setting {} small singular values to zero'.format(small.sum()))
            z[w <= thresh] = 0.0
        p = np.dot(V_T.T, np.dot(np.diag(z), np.dot(U.T,y)))   
    
    # compute best-fit model
    if star_basis_type == 'eye':
        Star = p[:K]
        L = K
    else:
        a = p[:L]
        Star = np.dot(a,A)
    Sys = np.zeros((J,K))
    for j in range(J):
        b = p[L+j*N:L+(j+1)*N]
        if T is not None:
            B = T[j,:,:].T
        Sys[j,:] = np.dot(b,B)

    # compute per-camera and averaged corrected LCs
    Fc = F - Sys
    w = 1./sigma**2
    w_av = np.mean(w, axis=1)
    Fc_av = np.average(Fc, axis=0, weights = w_av)
    
    return Fc, Fc_av, Star, Sys, p

