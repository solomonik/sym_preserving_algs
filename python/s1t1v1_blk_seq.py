import numpy as np

def naive_gemm(A,B,n,b):
    C = np.dot(A,B)
    C += C.reshape((n,b,n,b)).transpose([0,3,2,1]).reshape((n*b,n*b))

def fast_gemm(A,B,n,b):
    A = A.reshape((n,b,n,b)).transpose([0,2,1,3])
    B = B.reshape((n,b,n,b)).transpose([0,2,1,3])
    sA = np.einsum("ijxy->ixy",A)
    sB = np.einsum("ijxy->ixy",B)
    for i in range(n):
        sA[i,:,:] -= 2.*A[i,i,:,:]
        sB[i,:,:] -= 2.*B[i,i,:,:]
    nn = n*(n-1)*(n-2)/6
    idPC = np.zeros((nn,n,n))
    idPA = np.zeros((nn,n,n))
    idPB = np.zeros((nn,n,n))
    l = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                idPC[l,i,j] = 1
                idPA[l,i,k] = 1
                idPB[l,j,k] = 1
                l++
    oA = np.einsum("aij,ijxy->axy",idPC,A)+np.einsum("aik,ikxy->axy",idPA,A)+np.einsum("ajk,jkxy->axy",idPB,A)
    oB = np.einsum("aij,ijxy->axy",idPC,B)+np.einsum("aik,ikxy->axy",idPA,B)+np.einsum("ajk,jkxy->axy",idPB,B)
    P = np.einsum("axy,ayz->axz",oA,oB)
    Z = np.einsum("aij,axy->ijxy",idP,P)
    Z = np.einsum("aij,axy->ijxy",idPC,P) + np.einsum("aik,axy->ikxy",idPA,P) + np.einsum("ajk,axy->jkxy",idPB,P)
                
