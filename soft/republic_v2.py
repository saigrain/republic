import numpy as np
import matplotlib.pylab as plt

def cosine_func(p,x):
  return np.cos(x * p[0] + p[1])

def exp2_func(p,x):
  return np.exp(-(x-p[1])**2 + p[1])

def mk_basis(t, K, basis = 'cos', include_bias = True, doplot=True, normalise = True):
  N = len(t)
  tmin, tmax = np.nanmin(t), np.nanmax(t)
  T = tmax - tmin
  dt = np.nanmedian(t[1:]-t[:-1])
  if basis == 'fourier':
    A = np.zeros((K*2,N))
    for k in range(K): 
      arg = (k+1) * (np.pi/2) * (t / T) 
      A[2*k,:] = np.cos(arg)
      A[2*k+1,:] = np.sin(arg)
  elif basis == 'gauss':
    sig = T/(K-3)
    mus = np.linspace(tmin-sig,tmax+sig,K)
    A = np.zeros((K,N))
    for k in range(K):
      arg = (t-mus[k])/sig
      A[k,:] = np.exp(-arg**2)
  elif basis == 'poly':
    A = np.zeros((K,N))
    arg = (t-tmin)/T
    for k in range(K):
      A[k,:] = arg**(k+1)
  elif basis == 'tanh':
    sig = T/(K-3)
    mus = np.linspace(tmin-sig,tmax+sig,K)
    A = np.zeros((K,N))
    for k in range(K):
      arg = (t-mus[k])/sig
      A[k,:] = np.tanh(arg)
  else:
    print('ERROR: Basis function type {} not recognised'.format(basis))
    return
  if normalise:
    for k in range(K):
      tmp = A[k,:]
      tmp -= tmp.mean()
      A[k,:] = tmp / tmp.std()
  if include_bias:
    A = np.concatenate([np.ones_like(t).reshape((1,N)),A])
  if doplot:
    fig,ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(t, A.T,lw=0.5)
    ax[0].set_title('{} basis'.format(basis))
    ax[0].set_ylabel('basis')
    for i in range(10):
      k = A.shape[0]
      w = np.random.uniform(-1/k,1/k,k)
      ax[1].plot(t, np.dot(w,A))
    ax[1].set_ylabel('examples')
    ax[1].set_xlabel('input var.')
    ax[1].set_xlim(t.min(),t.max())
    plt.show()
  return A

def mk_segmented_basis(t,s, K, basis = 'poly', include_bias = True, doplot=True):
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

# functions to compute signal given basis and coefficients
def get_Star(a,X):
  return np.dot(a, X)
def get_Sys(b,B):
  J,L = b.shape
  N = B.shape[-1]
  Sys = np.zeros((J,N))
  for j in range(J):
    Sys[j,:] = np.dot(b[j,:],B) 
  return Sys
def get_Model(p,A,B):
  K = A.shape[0]
  a = p[:K]
  L = B.shape[0]
  J = int((len(p)-K)/L)
  b = p[K:].reshape((J,L))
  return get_Star(a,A)[None,:] + get_Sys(b,B)

def simulate_dataset(N=1000,Nseg=3,J=4,K=6,L=3,star_basis='fourier',sys_basis='poly', typical_sig=0.1):
  # generate time array, and array of segment IDs
  t = np.linspace(0,10,N)
  s = np.zeros(N,'int')
  breaks = np.sort(np.random.randint(0,N,Nseg-1))
  for b in breaks:
    s[b:] += 1
  # stellar basis
  A = mk_basis(t, K, basis = star_basis, include_bias = False, doplot=False)
  K_actual = A.shape[0]
  a = np.random.uniform(-1,1,K_actual)
  Star = get_Star(a,A)
  # systematics basis and signal
  B = mk_segmented_basis(t, s, L-1, basis = sys_basis, include_bias = True, doplot=False)
  L_actual = B.shape[0]
  b = np.zeros((J,L_actual))
  for j in range(J):
    b[j,:] = np.random.uniform(-1,1,L_actual)
  Sys = get_Sys(b,B) 
  # combine
  p_true = np.concatenate([a,b.flatten()])
  M_true = get_Model(p_true,A,B)
  sig = np.exp(np.random.uniform(-np.log(2),np.log(2),J)) * typical_sig
  sigma = np.zeros((J,N)) + sig[:,None]
  WN = np.random.normal(0,1,(J,N)) * sigma
  F = M_true + WN
  return t, s, F, sigma, p_true, M_true, A, B, Star, Sys

def fit_model(t, s, F, sigma, K_in=6,L_in=3,star_basis='fourier',sys_basis='poly',doplot=False):
  J, N = F.shape
  # construct basis sets
  A = mk_basis(t, K_in, basis = star_basis, include_bias = False, doplot=False)
  B = mk_segmented_basis(t, s, L_in-1, basis = sys_basis, include_bias = True, doplot=False)
  K = A.shape[0]
  L = B.shape[0]
  P = K + J * L
  # construct design matrix
  X = np.zeros((P,P))
  X[:K,:K] = ((A[:,None,:] * A[None,:,:])[:,:,None,:] / (sigma[None,None,:,:])**2).sum(axis=-1).sum(axis=-1)
  for j in range(J):
    D = ((A[:,None,:] * B[None,:,:]) / sigma[j,:]**2).sum(axis=-1)
    X[:K,K+j*L:K+(j+1)*L] = D
    X[K+j*L:K+(j+1)*L,:K] = D.T
    E = ((B[:,None,:] * B[None,:,:]) / sigma[j,:]**2).sum(axis=-1)
    X[K+j*L:K+(j+1)*L,K+j*L:K+(j+1)*L] = E
  y = np.zeros(P)
  y[:K] = (A[:,None,:] * (F / sigma**2)[None,:,:]).sum(axis=-1).sum(axis=-1)
  for j in range(J):
    y[K+j*L:K+(j+1)*L] =  (B * (F / sigma**2)[None,j,:]).sum(axis=-1)
  # regularise (ridge regression)
#  l1 = 1./K
  X += np.eye(P)# * l1
#  l2 = 1./(J*L) 
#  X[K:,K:] += np.eye(J*L) * l1
  # solve for coefficients
  from numpy.linalg import svd
  from scipy.linalg import cho_solve, cho_factor
  U, w, V_T = svd(X,hermitian=True)
  z = 1. / w
  w_max = abs(w).max()
  thresh = 1e-12 * w_max
  small = w <= thresh
  if small.any():
    print('Setting {} small singular values to zero'.format(small.sum()))
    z[w <= thresh] = 0.0
  p_fit = np.dot(V_T.T, np.dot(np.diag(z), np.dot(U.T,y)))
  M_fit = get_Model(p_fit, A, B)
  Star_fit = get_Star(p_fit[:K], A)
  pp = np.copy(p_fit)
  pp[:K] = 0
  Sys_fit = get_Model(pp, A, B)
  if doplot:
    fig,ax = plt.subplots(4,1,sharex=True, figsize = (12,12))
    for ss in np.unique(s):
      sv = max(t[s==ss])
      for a in ax:
        a.axvline(sv,c='k',ls=':',alpha=0.5)
    ax[0].set_ylabel('observed')
    for j in range(F.shape[0]):
      ax[0].plot(t, F[j,:], 'C{}.'.format(j%10), ms=4,alpha=0.5)
    ax[0].plot(t,M_fit.T,'k-')
    ax[1].set_ylabel('systematics-corrected')
    for j in range(F.shape[0]):
      ax[1].plot(t, F[j,:]-Sys_fit[j,:], 'C{}.'.format(j%10), ms=4,alpha=0.5)
    ax[1].plot(t, Star_fit, 'k-')
    ax[2].set_ylabel('star-corrected')
    for j in range(F.shape[0]):
      ax[2].plot(t, F[j,:]-Star_fit, 'C{}.'.format(j%10), ms=4,alpha=0.5)
    ax[2].plot(t, Sys_fit.T, 'k-')
    ax[3].set_ylabel('residuals')
    res = F - M_fit
    for j in range(F.shape[0]):
      ss = res[j,:].std()
      if j == 0:
        offset = 0
      else:
        offset -= 5 * ss
      ax[3].plot(t, res[j,:]+offset, 'C{}.'.format(j%10), ms=4,alpha=0.5)
      ax[3].axhline(offset, color = 'k')
      offset -= 5 * ss
    ax[3].set_xlim(t.min(),t.max())
    ax[3].set_xlabel('time')
    plt.tight_layout()
  return {'par': p_fit, 'M_fit': M_fit, 'A': A, 'B': B, 'Star_fit': Star_fit, 'Sys_fit': Sys_fit}, fig
  
def read_LCs_james(star_id = 0):
  lcdir = '../sim/set1_20210330/'
  import glob as g
  for c in range(24):
    c_files = np.sort(g.glob(lcdir+'c{:02d}_*_lcs.txt'.format(c+1)))  
    for i,fl in enumerate(c_files):
      if i == 0:
        t, f = np.loadtxt(fl, usecols=[0, star_id+1], unpack=True)
        s = np.zeros_like(t,'int')
      else:
        tt, ff = np.loadtxt(fl, usecols=[0, star_id+1], unpack=True)
        t = np.concatenate([t,tt])
        f = np.concatenate([f,ff])
        s = np.concatenate([s,np.zeros_like(tt,'int')+i])
    sig = np.median(abs(f[1:]-f[:-1]))
    if c == 0:
      N = len(f)
      F = f.reshape((1,N))
      sigma = np.zeros_like(F) +sig
    else:
      F = np.concatenate([F,f.reshape((1,N))])
      sigma = np.concatenate([sigma,np.zeros((1,N))+sig])
  # normalise
  m = F.mean()
  F = 1e3 * (F/m-1) # in parts per thousand
  sigma = 1e3 * sigma / m
  t /= 86400 # in days
  return t, s, F, sigma

#t, s, F, sigma, p_true, M_true, A, B, Star, Sys = simulate_dataset(star_basis='gauss',K=10)

for star_id in range(36):
  print(star_id)
  t, s, F, sigma = read_LCs_james(star_id)
  r,fig = fit_model(t, s, F, sigma, star_basis ='fourier', K_in = 50, L_in = 4, doplot=True)
  plt.savefig('../sim/set1_20210330/republic2_plots/star_{:02}.png'.format(star_id))
  plt.close('all')