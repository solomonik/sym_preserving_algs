import numpy as np
import sys
import time
import argparse
import time
from pathlib import Path
from os.path import dirname, join

def naive_gemm(A,B,n,b):
    C = np.dot(A,B)
    C += C.reshape((n,b,n,b)).transpose([2,1,0,3]).reshape((n*b,n*b))
    return C

def fast_gemm(A,B,n,b):
    A = A.reshape((n,b,n,b)).transpose([0,2,1,3])
    B = B.reshape((n,b,n,b)).transpose([0,2,1,3])
    sA = -np.einsum("ijxy->ixy",A)
    sB = -np.einsum("ijxy->ixy",B)
    for i in range(n):
        sA[i,:,:] += 2.*A[i,i,:,:]
        sB[i,:,:] += 2.*B[i,i,:,:]
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
    Z += Z.transpose([1,0,2,3])
    U = np.einsum("ijxy,iyz->ijxz",A,sB)
    U += U.transpose([1,0,2,3])
    V = np.einsum("ixy,ijyz->ijxz",sA,B)
    V += V.transpose([1,0,2,3])
    W = np.einsum("ijxy,ijyz->ijxz",A,B)
    sW = np.einsum("ijxy->ixy",W)
    for i in range(n):
        sW[i,:,:] -= W[i,i,:,:]
        U[i,i,:,:] = 0
        V[i,i,:,:] = 0

    #print(U.transpose([0,2,1,3]).reshape((n*b,n*b)))
    #print(V.transpose([0,2,1,3]).reshape((n*b,n*b)))
    #print("W",W.transpose([0,2,1,3]).reshape((n*b,n*b)))
    #print((- sW.reshape((1,n,b,b)) - sW.reshape((n,1,b,b))).transpose([0,2,1,3]).reshape((n*b,n*b)))
    C = Z - (n-8)*W + U + V - sW.reshape((1,n,b,b)) - sW.reshape((n,1,b,b))
    for i in range(n):
        C[i,i,:,:] = 2*sW[i,:,:] + 2*W[i,i,:,:]
    return C.transpose([0,2,1,3]).reshape((n*b,n*b))

def test(n,b):
    A = np.ones((n,b,n,b))
    B = np.ones((n,b,n,b))
    np.random.seed(42)
    A = np.random.random((n,b,n,b))
    B = np.random.random((n,b,n,b))
    A = (A + A.transpose([2,1,0,3])).reshape((n*b,n*b))
    #A = np.ones((n*b,n*b))
    B = (B + B.transpose([2,1,0,3])).reshape((n*b,n*b))
    #B = np.ones((n*b,n*b))
    Ac = A.copy()
    Bc = B.copy()
    C_naive = naive_gemm(A,B,n,b)
    C_fast = fast_gemm(Ac,Bc,n,b)
    #print("Naive method yields")
    #print(C_naive)
    #print("Fast method yields")
    #print(C_fast)
    #print("Ratio is")
    #print(C_fast/C_naive)
    err = np.linalg.norm(C_naive-C_fast)/np.linalg.norm(C_naive)
    print("n=",n,"b=",b)
    print("Relative two norm error is",err)


def bench(n,b,niter):
    A = np.ones((n,b,n,b))
    B = np.ones((n,b,n,b))
    np.random.seed(42)
    A = np.random.random((n,b,n,b))
    B = np.random.random((n,b,n,b))
    A = (A + A.transpose([2,1,0,3])).reshape((n*b,n*b))
    B = (B + B.transpose([2,1,0,3])).reshape((n*b,n*b))
    Ac = A.copy()
    Bc = B.copy()
    times_naive = []
    times_fast = []

    print("Benchmarking naive for",niter,"iterations")
    for i in range(niter):
        t0 = time.time()
        C_naive = naive_gemm(A,B,n,b)
        t1 = time.time()
        ite1 = t1 - t0
        print(ite1)
        times_naive.append(ite1)

    print("Benchmarking fast for",niter,"iterations")
    for i in range(niter):
        t0 = time.time()
        C_fast = fast_gemm(Ac,Bc,n,b)
        t1 = time.time()
        ite1 = t1 - t0
        print(ite1)
        times_fast.append(ite1)

    avg_naive = np.mean(times_naive)
    avg_fast = np.mean(times_fast)

    print("Average time for naive is",avg_naive)
    print("Average time for fast is",avg_fast)

    stddev_naive = np.std(times_naive)
    stddev_fast = np.std(times_fast)

    print("95% confidence interval for naive is [",avg_naive - 2*stddev_naive,",",avg_naive + 2*stddev_naive,"]")
    print("95% confidence interval for fast is [",avg_fast - 2*stddev_fast,",",avg_fast + 2*stddev_fast,"]")


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
    parser.add_argument(
        '--niter',
        type=int,
        default=10,
        metavar='int',
        help='Number of iterations for benchmarking (default: 10)')

    args, _ = parser.parse_known_args()
    n = args.n
    b = args.b
    niter = args.niter
    test(n,b)
    bench(n,b,niter)

