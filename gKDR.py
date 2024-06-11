import numpy

def MedianDist(X):
    N = X.shape[0]
    ab = X * X.transpose(0, 1) 
    aa = numpy.diag(ab)
    Dx = aa.repeat(1,N) + aa.transpose(0, 1).repeat(N, 1) - 2 * ab
    Dx = Dx- numpy.diag(numpy.diag(Dx))
    dx=numpy.nonzero(numpy.reshape(Dx,N*N,1))
    return numpy.sqrt(numpy.median(dx))

# SQDIST - computes squared Euclidean distance matrix
#          computes a rectangular matrix of pairwise distances
# between points in A (given in columns) and points in B

# NB: very fast implementation taken from Roland Bunschoten

def sqdist(a,b):
    aa = numpy.sum(numpy.dot(a,a),1)
    bb = numpy.sum(numpy.dot(b,b),1)
    ab = numpy.dot(a.transpose(0,1), b)
    d = numpy.abs(numpy.repeat(aa.transpose(0,1), [1, bb.shape[2]])+
                  numpy.repeat(bb,[aa.shape[2],1]) - 2*ab)
    return d

# CHOL_INC_FUN - incomplete Cholesky decomposition of the Gram matrix defined
#                by data x, with the Gaussiab kernel with width sigma
#                Symmetric pivoting is used and the algorithms stops 
#                when the sum of the remaining pivots is less than TOL.

# CHOL_INC returns returns an uPvecer triangular matrix G and a permutation 
# matrix P such that P'*A*P=G*G'.

# P is ONLY stored as a reordering vector PVEC such that 
#                    A(Pvec,Pvec)= G*G' 
# consequently, to find a matrix R such that A=R*R', you should do
# [a,Pvec]=sort(Pvec); R=G(Pvec,:);

# Copyright (c) Francis R. Bach, 2002.

def chol_inc_gauss(x,sigma,tol):
   n = x.shape[2] 
   Pvec = numpy.linspace(1, n, 1)
   I = []
   # calculates diagonal elements (all equal to 1 for gaussian kernels)
   diagG=numpy.ones((n,1))
   i = 1 
   G = []
   while((sum(diagG[i:n])>tol)):
    G = [G,numpy.zeros((n,1))]
    # find best new element
    if i>1 :
        _, jast=numpy.max(diagG[i:n])
        jast=jast+i-1
        # updates permutation
        Pvec[[i, jast]] = Pvec[[jast,i]]
        # updates all elements of G due to new permutation
        G[[i,jast],1:i]=G[[ jast,i],1:i]
        # do the cholesky update
    else: 
       jast = 1
    
    G[i,i]=numpy.sqrt(diagG[jast]) #A(Pvec(i),Pvec(i));
    if i<n :
    # calculates newAcol=A(Pvec((i+1):n),Pvec(i))
        newAcol = numpy.exp(-.5/sigma^2*sqdist(x[:, Pvec[(i+1):n]]),x[:,Pvec[i]])
        if (i>1):
            G[(i+1):n,i]=1/G[i,i]*( newAcol - G[(i+1):n,1:(i-1)]*G[i,1:(i-1)].transpose())
        else:
            G[(i+1):n,i]=1/G[i,i] * newAcol;
	#updates diagonal elements
    if i<n: 
        diagG[(i+1):n]=numpy.ones((n-i,1))-numpy.sum( numpy.dot(G[(i+1):n,1:i],G[(i+1):n,1:i] ),2 );
   i=i+1
   return G, Pvec



#-----------------------------------------------
# KernelDeriv() based on KernelDeriv() except using incomplete cholesky approximation to
# solve memory problem.
#
# Arguments
#  X:  explanatory variables (input data)
#  Y:  response variables (teaching data)
#  K:  dimension of effective subspaces
#  SGX:  bandwidth (deviation) parameter in Gaussian kernel for X
#  SGY:  bandwidth (deviation) parameter in Gaussian kernel for Y
#  EPS:  regularization coefficient
#
# Return value(s)
#  B:  orthonormal column vectors (M x K)
#  t:  value of the objective function
#  R:  estimated Mn (added by YI)
# 
#-----------------------------------------------

def KernelDeriv_chol(X,Y,K,SGX,SGY,EPS):
	N,M=numpy.shape(X)
	tol=0.000001   # tolerance for incomplete cholesky approximation
	I=numpy.eye(N)

    # Calculate Gram matrix of X
	sx2=2*SGX*SGX
	ab=X*X.transpose(0,1)
	aa=numpy.diag(ab)
	D=numpy.repeat(aa,1,N)
	xx=numpy.max(D + D.transpose(0,1) - 2*ab, numpy.zeros(N,N))
	Kx=numpy.exp(numpy.div(-xx,sx2))

	# Calculate Gram matrix of Y
	sy2=2*SGY*SGY
	ab=Y*Y.transpose(0,1)
	aa=numpy.diag(ab)
	D=numpy.repeat(aa,1,N)
	yy=numpy.max(D +D.transpose(0,1) - 2*ab, numpy.zeros(N,N))
	Ky=numpy.exp(numpy.div(-yy,sy2))
	
    #incomplete cholesky approximation of Ky
	print('Execute chol_inc_gauss')
	G, Pvec = chol_inc_gauss(Y.transpose(0,1),SGY,tol)
	a, Pvec = numpy.sort(Pvec)
	Ry = G[Pvec,:]
	r = numpy.length(Ry[1,:])
	Ty=numpy.div(Ry.transpose(0,1), (Kx+numpy.dot(numpy.matmul(N,EPS), numpy.eye(N))))
	
    # Derivative of k(X_i, x) w.r.t. x
	print("Derivative of k(X_i, x")
	Dx=numpy.reshape(numpy.repeat(X,N,1),(N,N,M))
	Xij=Dx-numpy.permute(Dx,(2, 1, 3))
	Xij=numpy.div(Xij,SGX)/SGX
	H=numpy.dot(Xij, numpy.repeat(Kx,(1,1,M)))
	print('Finished Derivative of k(X_i, x')
    
    # compute the matrix for gKDR
	Hy=numpy.reshape(Ty*numpy.reshape(H,(N,N*M)), (r*N,M))
	R=numpy.matmul(Hy.transpose(0,1), Hy)
	V,L = numpy.linalg.eig(R)
	e,idx=numpy.sort(numpy.diag(L),descending=True)
	B=V[:,idx[1:K]]
	t=numpy.sum(e[idx[1:K]])
	return B, R, t, Kx