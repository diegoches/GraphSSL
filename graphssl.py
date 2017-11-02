#! /usr/bin/env python
# -*- coding: iso-8859-1 -*-
#
#  make2Dpts.py
#
#  create 2D point sets, generate plots, and generate sparse matrices for
#    experiments
#

import math
import numpy as np
import numpy.linalg as npla
import scipy.linalg as spla
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spspla
import scipy.io as spio
import scipy.optimize as spop
import matplotlib.mlab as mlab
import itertools

"""
Notes: 
 
 three # , means commented by Diego

"""


def graphIntegration (xl, k, M, xi = None):
  sm = M.sum(0)
  n = sm.shape[1]
  if spsp.issparse(xl):
    b = xl.todense()
  else:
    b = np.array(xl)
  A = spsp.spdiags(sm,[0],n,n) - M + k*spsp.spdiags(map(abs,b),[0],n,n)

  return spspla.spsolve(A,b)


def graphIntegration2 (xl, k, M, xi = None):
  sm = M.sum(0)
  n = sm.shape[1]
  if spsp.issparse(xl):
    b = xl.todense()
  else:
    b = np.array(xl)

    
  A = spsp.spdiags(sm,[0],n,n) - M
  b2 = A * b

  return spspla.spsolve(A,b2)

def graphIntegration3 (xl, k, M, xi = None):
  sm = M.sum(0)
  n = sm.shape[1]
  if spsp.issparse(xl):
    b = np.array(xl.todense())
  else:
    b = np.array(xl)

  axl = np.abs(xl)
  naxl = 1 - axl

  A = spsp.spdiags(naxl,[0],n,n) * (spsp.spdiags(sm,[0],n,n) - M) + spsp.spdiags(axl,[0],n,n)

  b2 = A * b

  b3 = np.select([naxl,axl],[b2,xl] )

  print A.todense()
  print b3

  return spspla.spsolve(A,b3)


def FIRGraph( f, L, x):
  """
  Applies the FIR Graph filter over signal x.

  f:  np.matrix (column vector) of coefficients of filter
      (f[0]*I + ... f[k-1]*L^k-1)
  L:  The laplacian of the Graph
  x:  Input signal on the vertices.

  Returns f(L)*x   ( f[0]*x + f[1]*L*x + ... + f[k-1] * L^k-1 * x)
  """
  accm = spsp.csr_matrix(L.shape)
  n = L.shape[0]

  for fi in reversed(range(f.shape[0])):
    accm = L * accm + f[fi,0] * spsp.identity( n )

  return accm * x

def fastFIRGraph( f, L, x):
  """
  Applies the FIR Graph filter over signal x.

  Only performs Matrix - Vector multiplications.


  f:  np.matrix (column vector) of coefficients
      (f[0]*I + ... f[k-1]*L^k-1)
  L:  The laplacian of the Graph
  x:  the input signal on the vertices, a single column matrix

  Returns f[0,0]*x + f[1,0]*L*x ... f[k-1,0]*L^k-1* x
  """
  x2 = np.matrix(x)

  if x2.shape[1] != 1:
    x2 = x2.T

  accm = spsp.csr_matrix(x2.shape)

  for fi in reversed(range(f.shape[0])):
    accm = L * accm + f[fi,0] * x2

  return accm


def powersOfL (k, L):
  """
  Calculate the 0 - k powers of L and returns them in a list
  """

  accm = spsp.identity(L.shape[0])
  lpowers = [ accm.copy() ]

  for _ in range(k-1):
    accm = accm * L
    lpowers.append(accm.copy())

  return lpowers


def calcKrylovVectors (L, x, k):
  #  n = len(x)

  kvecs = np.matrix(x)

  if kvecs.shape[1] != 1:
    kvecs = kvecs.T

  for i in range(1,k):
    kvecs = np.hstack((kvecs, L*kvecs[:,-1]))

  return kvecs


def calcMatAnC (L, M, l, k):
  """
  Calculates matrices A and C for the optimum FIR
  coefficient calculation.


  L:  Laplacian of the Graph (sparse matrix nxn).
  M:  Similarity matrix of the graph (sparse matrix nxn).
  l:  Label vector where 1 or -1 classes, 0 unlabeled (nx1 matrix).
  k:  Filter size.

  returns (A, C, cl)
  """

#  K = len(lpowers)
  A = np.matrix(np.zeros( (k, k) ))

  numConst = abs(l).sum()
  C = np.matrix(np.zeros( (k, numConst )))
  cl = np.transpose(np.matrix(np.empty( numConst )))

  #  L = lpowers[1].tocoo()
  #  Ll = lambda m: m*l
  # map(Ll, lpowers)
  
  pl = calcKrylovVectors(L, l, k)
  ###spio.savemat('krylov.mat', { 'pl':pl } )

  L2 = M.tocoo()
  for i,j,v in itertools.izip(L2.row, L2.col, L2.data):
    lambij = np.empty((1,k))
    for ki in range(k):
      lambij[0,ki] = pl[i,ki] - pl[j,ki]
    A = A  + v * (lambij * np.transpose(lambij)) #  (i,j,v)

  idxc = 0
  for i in range(len(l)):
    if abs(l[i,0]) > 0.1:
      # for ki in range(k):
      #   C[ki,idxc] = pl[i,ki]
      C[:,idxc] = pl[i,:].T
      cl[idxc,0] = l[i,0]
      idxc = idxc + 1


  return A, C, cl


def calcFilterCoef ( L, W, l, k, w=0.5 ):
  """
  Calculates filter coeficients.

  L:  Laplacian of the Graph (sparse matrix nxn).
  W:  Similarity matrix of the graph (sparse matrix nxn).
  l:  Label vector where 1 or -1 classes, 0 unlabeled (nx1 matrix).
  k:  Filter size.
  w:  (0, 1) relative weight between smoothness (0) and constraints (1)

  returns (A, C, cl)
  """
  f = np.empty(k)

  #  lpowers =  powersOfL (k, L)

  lmat = np.matrix(np.array(l).T)

  A, C, cl = calcMatAnC (L, W, lmat ,k) 

  wa = (1-w) / W.nnz
  wc = w / len(cl)

  M = wa * A   + wc * ( C * np.transpose(C) )
  b = wc * C * cl

  if False:###True:
    print "matrix M:"
    np.set_printoptions(4.2,suppress=True)
    print M 

    print "eigenvalues of M: "
    np.set_printoptions(12.8,suppress=True)
    print spla.eigvals(M).real
    print "done eigen"

    spio.savemat('matsACM.mat', {'A':A, 'C':C, 'M':M})

  ###f = npla.solve(M,b)
  f,d_re,d_ra,d_s = npla.lstsq(M,b)

  np.set_printoptions(4.2,suppress=True)
  ###print L.nnz, len(cl), wa, wc

  #np.set_printoptions(formatter={'all':lambda x: "%4.2f" % x})
 
  ###print f.T
  
  return f

def firGraphEnergy( lb, L, x, t):

  msk = abs(t)

  accm = spsp.csr_matrix(L.shape)
  n = l.shape[0]

  for fi in reversed(lb):
    accm = L * accm + fi * spsp.spdiags( [ [fi]*n ],[0],n,n )

  return npla.norm( (accm * x - t) * msk )


def firGraphEnergyGrad( lb, L, x, t):

  msk = abs(t)

  n = l.shape[0]

  o = np.zeros(n)
  grd = np.zeros(len(lb))

  pL = spsp.identity(n)
  accm = spsp.csr_matrix(L.shape)

  for i in range(lb):
    #    accm = L * accm + fi * spsp.spdiags( [ [fi]*n ],[0],n,n )
    accm = accm + pL * lb[i]
    grad[i] = npla.norm( (pL * x - t ) * msk )
    pL = L * pL

  return npla.norm( (accm * x - t) * msk), grad


def rowNormLaplacian(M):
  """
  Calculates the symetric normalized Laplacian.
  D2 * ( D - M )

  M: the similarity matrix.

  returns the row normalized Laplacian.
  """
  sm = M.sum(0)
  sm2 = 1. / sm
  n  = sm.shape[1]
  D2 = spsp.spdiags(sm2,[0],n,n)
  L  = spsp.identity(n) - D2 * M 

  return L

def symNormLaplacian(M):
  """
  Calculates the symetric normalized Laplacian.
  D2 * ( D - M ) * D2

  M: the similarity matrix.

  returns the normalized symmetric Laplacian.
  """
  sm = M.sum(0)
  sm2 = 1./np.sqrt(sm)
  n  = sm.shape[1]
  #  D  = spsp.spdiags(sm,[0],n,n)
  D2 = spsp.spdiags(sm2,[0],n,n)
  L  = spsp.identity(n) - D2 * M * D2

  # acc = 0
  # for i in range(L.shape[0]):
  #   acc = acc + L[i,i]

  # print "\n\ntrace of L = %f \n\n" % acc

  return L
  
def rawLaplacian(M):
  """
  Calculates the unnormalized Laplacian.
    D - M 

  M: the similarity matrix.

  returns the unnormalized symmetric Laplacian.
  """
  sm = M.sum(0)
  n  = sm.shape[1]
  return spsp.spdiags(sm,[0],n,n) - M


def graphSignalProcessingSSL(xl, k, M, xi = None, w=0.5):
  """
  Calculates optimum filter coefficients and process an input signal.

  xl:  Labels of a few of the vertices (0 unlabeled, +-1 for classes)
  k :  Filter size
  M :  Symilarity matrix
  xi:  Input signal (if None, uses xl)
  w :  Weight of Smoothness (0) versus Labels (1)

  returns the filtered labels of the graph
  """
  
  L = symNormLaplacian(M)

  f = calcFilterCoef(L, M, np.array(xl), k, w )

  if xi == None:
    xi = xl

  return fastFIRGraph( f, L, xi)



if __name__ == "__main__":

  print "Module that provides semi-supervised graph methods."
