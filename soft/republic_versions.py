#!/usr/bin/env python
# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv, cho_solve, cho_factor, svd, solve_triangular
from numpy.linalg import qr
import republic_v1 as rv1

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

def mk_basis(t, M, basis = 'fourier', include_bias = True, doplot=True, normalise = True):
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
        A = np.eye(K,dtype=np.float128)
        normalise = False
        include_bias = False
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
        np.random.seed(1234)
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

def mk_segmented_basis(t, s, K, basis = 'poly', include_bias = True, normalise = True, doplot=True):
    N = len(t)
    su = np.unique(s)
    Ns = len(su)
    for i,ss in enumerate(su):
        l = s == ss
        ts = t[l]
        As = mk_basis(t[l], K, basis = basis, include_bias = include_bias, normalise = normalise, doplot=False)
        if i == 0:
            A = np.zeros((As.shape[0], N))
            A[:,l] = As 
        else:
            As_ = np.zeros((As.shape[0], N))
            As_[:,l] = As 
            A = np.concatenate([A,As_])
#        As = mk_basis(t[l], K, basis = basis, include_bias = include_bias, normalise = normalise, doplot=False)
#        l = s != ss
#        As[:,l] = 0
#        if i == 0:
#            A = np.copy(As)
#        else:
#            A = np.concatenate([A,As])
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
    print('No cameras:', J)
    print('No frames:', K)

    w = 1./sigma**2
    Fw = F * w
    w128 = w.astype(np.float128)
    Fw128 = Fw.astype(np.float128)
    
    # construct stellar basis
    print('Using basis type {} for stellar signal.'.format(star_basis_type))
    A = mk_basis(t, L_in, basis = star_basis_type, include_bias = False, doplot=False) # transpose of matrix U in model_v2.ipynb
    A128 = A.astype(np.float128)
    stellar_basis_dim = A.shape[0]
    print('No free params for stellar signal', stellar_basis_dim)
    if star_basis_type == 'eye':
        C = np.diag(w.sum(axis=0))
        Cinv = np.diag(1./w.sum(axis=0))
        y1 = Fw.sum(axis=0)
    else:
        y1 = np.zeros(stellar_basis_dim) # vector q in model_v2.ipynb
        C = np.zeros((stellar_basis_dim, stellar_basis_dim))
        for m1 in range(stellar_basis_dim):
            y1[m1] = ((A128[m1,:])[None,:] * Fw128).sum().astype(np.float)
            for m2 in range(stellar_basis_dim):
                C[m1,m2] = ((A128[m1,:] * A128[m2,:])[None,:] * w128).sum()
        # check condition number of C
        print('Condition no. of C:', np.linalg.cond(np.dot(C.T,C)))
        Cinv = inv(C)

    if T is None:
        print('Using functional basis for systematics')
        print('Basis type:', sys_basis_type)
        if s is None:
            s = np.zeros(K).astype(int)
        B = mk_segmented_basis(t, s, N_in, basis = sys_basis_type, include_bias = True, doplot=False)
        N = B.shape[0]
        T = np.zeros((J,K,N))
        for j in range(J):
            T[j,:,:] = B.T           
    else:
        print('Using user-supplied systematic basis')
    T128 = T.astype(np.float128)
    N = T.shape[-1]
    sys_basis_dim = J * N
    print('No trends per camera:', N)
    print('No free params for systematics:', sys_basis_dim)
    y2 = np.zeros(sys_basis_dim) # vector r in model_v2.ipynb 
    D = np.zeros((stellar_basis_dim, sys_basis_dim))
    E = np.zeros((sys_basis_dim, sys_basis_dim))
        
    for j in range(J):
        for n1 in range(N):
            y2[j*N+n1] = (Fw128[j,:] * T128[j,:,n1]).sum() # vector r_j in model_v2.ipynb
            for m in range(stellar_basis_dim):
                D[m,j*N+n1] = (A128[m,:] * T128[j,:,n1] * w128[j,:]).sum() # matrix D_j in model_v2.ipynb 
            for n2 in range(N):     
                E[j*N+n1,j*N+n2] = (T128[j,:,n1] * T128[j,:,n2] * w128[j,:]).sum() # matrix E in model_v2.ipynb 
    # check condition number of E
    print('Condition no. of E:', np.linalg.cond(np.dot(E.T,E)))

    X = np.zeros((stellar_basis_dim+sys_basis_dim,stellar_basis_dim+sys_basis_dim))
    X[:stellar_basis_dim,:stellar_basis_dim] = C
    X[:stellar_basis_dim,stellar_basis_dim:] = D
    X[stellar_basis_dim:,:stellar_basis_dim] = D.T
    X[stellar_basis_dim:,stellar_basis_dim:] = E
    print('Condition no. of X:', np.linalg.cond(np.dot(X.T,X)))    
    plt.figure()
    plt.imshow(X)
    plt.colorbar()
    
    y = np.zeros(stellar_basis_dim+sys_basis_dim)
    y[:stellar_basis_dim] = y1
    y[stellar_basis_dim:] = y2
    
    # block-wise inversion of full design matrix
    CinvD = np.dot(Cinv, D)
    DTCinvD = np.dot(D.T, CinvD)
    E_DTCinvD = E - DTCinvD
    # check condition number of E_DTCinvD
    print('Condition no. of E_DTCinvD:', np.linalg.cond(np.dot(E_DTCinvD.T,E_DTCinvD)))
    E_DTCinvD_inv = inv(E_DTCinvD)
    bottomright = E_DTCinvD_inv
    topright = - np.dot(Cinv,np.dot(D,E_DTCinvD_inv))
    bottomleft = topright.T
    topleft = Cinv + np.dot(Cinv, np.dot(D, -bottomleft))
    Xinv = np.zeros((stellar_basis_dim+J*N,stellar_basis_dim+J*N))
    Xinv[:stellar_basis_dim,:stellar_basis_dim] = topleft
    Xinv[:stellar_basis_dim,stellar_basis_dim:] = topright
    Xinv[stellar_basis_dim:,:stellar_basis_dim] = bottomleft
    Xinv[stellar_basis_dim:,stellar_basis_dim:] = bottomright

#     X = np.zeros((stellar_basis_dim+sys_basis_dim,stellar_basis_dim+sys_basis_dim))
#     X[:stellar_basis_dim,:stellar_basis_dim] = C
#     X[:stellar_basis_dim,stellar_basis_dim:] = D
#     X[stellar_basis_dim:,:stellar_basis_dim] = D.T
#     X[stellar_basis_dim:,stellar_basis_dim:] = E
#     print('Condition no. of X:', np.linalg.cond(np.dot(X.T,X)))    
          
    # solve for coefficients
    # p = np.dot(Xinv,y)
    # c, low = cho_factor(X)
    # p = cho_solve((c, low), y)
    # p = solve(X,y)
    p = np.dot(Xinv,y)
    print(p[:10])
    print(p[-10:])
    
#     # compare to old republic_v1
#     if star_basis_type == 'eye':
#         _, _, _, interm = rv1.solve(F, T, sigma)
#         Cinv_, D_, E_, y_, CinvD_, DTCinvD_, E_DTCinvD_, E_DTCinvD_inv_, bottomright_, topright_, bottomleft_, topleft_, Xinv_, p_ = interm
#         for i in range(10):
#             print(p[i], p_[i])
#         for i in range(10):
#             print(p[-i], p_[-i])
            
# #         print('Checking Cinv:', np.allclose(Cinv,Cinv_))
#         plt.figure()
#         plt.imshow(Cinv-Cinv_)
#         plt.title('Cinv')
#         plt.colorbar()
#         print((Cinv-Cinv_).min(), (Cinv-Cinv_).max())
# #         print('Checking D:', np.allclose(D,D_))
#         plt.figure()
#         plt.imshow(D-D_)
#         plt.title('D')
#         plt.colorbar()
#         print((D-D_).min(), (D-D_).max())
# #         print('Checking E:', np.allclose(E,E_))
#         plt.figure()
#         plt.imshow(E-E_)
#         plt.title('E')
#         print((E-E_).min(), (E-E_).max())
#         plt.colorbar()
#         plt.figure()
#         plt.plot(y-y_)
#         plt.title('y')
#         print((y-y_).min(), (y-y_).max())
# #         print('Checking y:', np.allclose(y,y_))
# #         print('Checking CinvD:', np.allclose(CinvD,CinvD_))
# #         print('Checking DTCinvD:', np.allclose(DTCinvD,DTCinvD_))
# #         print('Checking E_DTCinvD:', np.allclose(E_DTCinvD,E_DTCinvD_))
# #         print('Checking E_DTCinvD_inv:', np.allclose(E_DTCinvD_inv,E_DTCinvD_inv_))
# #         print('Checking bottomright:', np.allclose(bottomright,bottomright_))
# #         print('Checking topright:', np.allclose(topright,topright_))
# #         print('Checking bottomleft:', np.allclose(bottomleft,bottomleft_))
# #         print('Checking topleft:', np.allclose(topleft,topleft_))
# #         print('Checking Xinv:', np.allclose(Xinv,Xinv_, atol=1e-10,rtol=1e-7))
# #         print('Checking p:', np.allclose(p,p_))
#         plt.figure()
#         plt.imshow(Xinv-Xinv_)
#         plt.title('Xinv')
#         plt.colorbar()
#         plt.figure()
#         plt.plot(p-p_)
#         plt.title('p')
              
#         residual_norm = np.linalg.norm(np.dot(gram,p)-np.dot(X.T,y))        
#         print('QR norm', residual_norm)

    
    # compute best-fit model
    if star_basis_type == 'eye':
        Star = p[:stellar_basis_dim]
    else:
        a = p[:stellar_basis_dim]
        Star = np.dot(a,A)
    Sys = np.zeros((J,K))
    for j in range(J):
        b = p[stellar_basis_dim+j*N:stellar_basis_dim+(j+1)*N]
        if T is not None:
            B = T[j,:,:].T
        Sys[j,:] = np.dot(b,B)

    # compute per-camera and averaged corrected LCs
    Fc = F - Sys
    w = 1./sigma**2
    w_av = np.mean(w, axis=1)
    Fc_av = np.average(Fc, axis=0, weights = w_av)
    
    return Fc, Fc_av, Star, Sys, p

