import numpy as np
import numpy.linalg as la

def fast_alg(A,b):
    n = b.shape[0]
    c = np.zeros(n,dtype=A.dtype)
    a = np.zeros(n,dtype=A.dtype)
    for i in range(n):
        a[i] = A[i,i]
        for j in range(i):
            z = A[i,j]*(b[i]+b[j])
            c[i] += z
            c[j] += z
            a[i] -= A[i,j]
            a[j] -= A[i,j]
    for i in range(n):
        c[i] += a[i]*b[i]
    return c

errs_ref  = []
errs_fast = []

n = 100

for logcnd in range(60): #np.logspace(0,60,60,base=2,dtype=int):
    A = np.asarray(np.random.random((n,n)),dtype=np.float64)
    [U,s,VT] = la.svd(A)
    s = 2.**np.linspace(0,logcnd,n)
    signs = ((np.random.random((n))>.5)-.5)*2
    s = s*signs
    A = U @ np.diag(s) @ U.T
    b = U[:,0]#np.asarray(np.random.random((n)),dtype=np.float64)
    c_ref = A @ b
    A = np.asarray(A,dtype=np.float32)
    b = np.asarray(b,dtype=np.float32)
    c = A @ b
    err_ref = la.norm(c_ref - c)/la.norm(c_ref)
    errs_ref.append(err_ref)
    print("Ref  err:",err_ref)
    c = fast_alg(A,b)
    err_fast = la.norm(c_ref - c)/la.norm(c_ref)
    errs_fast.append(err_fast)
    print("Fast err:",err_fast)
    print("Ratio is",err_fast/err_ref)
