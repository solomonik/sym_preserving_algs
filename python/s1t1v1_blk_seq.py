import numpy as np
import sys
import time
import argparse
from pathlib import Path
from os.path import dirname, join

def naive_gemm(A,B,n,b):
    C = np.dot(A,B)
    C += C.reshape((n,b,n,b)).transpose([2,1,0,3]).reshape((n*b,n*b))
    return C

def fast_gemm(A,B,n,b):
    A = A.reshape((n,b,n,b)).transpose([0,2,1,3])
    B = B.reshape((n,b,n,b)).transpose([0,2,1,3])
    sA = np.einsum("ijxy->ixy",A)
    sB = np.einsum("ijxy->ixy",B)
    for i in range(n):
        sA[i,:,:] -= 2.*A[i,i,:,:]
        sB[i,:,:] -= 2.*B[i,i,:,:]
    nn = n*(n-1)*(n-2)//6
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
                l+=1
    oA = np.einsum("aij,ijxy->axy",idPC,A)+np.einsum("aik,ikxy->axy",idPA,A)+np.einsum("ajk,jkxy->axy",idPB,A)
    oB = np.einsum("aij,ijxy->axy",idPC,B)+np.einsum("aik,ikxy->axy",idPA,B)+np.einsum("ajk,jkxy->axy",idPB,B)
    P = np.einsum("axy,ayz->axz",oA,oB)
    Z = np.einsum("aij,axy->ijxy",idPC,P) + np.einsum("aik,axy->ikxy",idPA,P) + np.einsum("ajk,axy->jkxy",idPB,P)
    U = np.einsum("ijxy,iyz->ijxz",A,sB) + np.einsum("ijxy,jyz->ijxz",A,sB)
    V = np.einsum("ixy,ijyz->ijxz",sA,B) + np.einsum("jxy,ijyz->ijxz",sA,B)
    W = np.einsum("ijxy,ijyz->ijxz",A,B)
    sW = np.einsum("ijxy->ixy",W)
    for i in range(n):
        sW[i,:,:] -= 2.*W[i,i,:,:]

    print(Z + (n-8)*W + U + V)
    print(sW.reshape((1,n,b,b)))
    C = Z + (n-8)*W + U + V - sW.reshape((1,n,b,b)) - sW.reshape((n,1,b,b))
    print(C)
    return C.transpose([0,2,1,3]).reshape((n*b,n*b))

def test(n,b):
    A = np.random.random((n,b,n,b))
    B = np.random.random((n,b,n,b))
    A = (A + A.transpose([2,1,0,3])).reshape((n*b,n*b))
    B = (B + B.transpose([2,1,0,3])).reshape((n*b,n*b))
    Ac = A.copy()
    Bc = B.copy()
    C_naive = naive_gemm(A,B,n,b)
    C_fast = fast_gemm(Ac,Bc,n,b)
    print(C_naive)
    print(C_fast)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n',
        type=int,
        default=4,
        metavar='int',
        help='Dimension of symmetric modes (default: 4)')
    parser.add_argument(
        '--b',
        type=int,
        default=10,
        metavar='int',
        help='Dimension of nonsymmetric modes (default: 10)')
    args, _ = parser.parse_known_args()
    n = args.n
    b = args.b
    test(n,b)

