#include <limits>
#include <stdlib.h>
#include <sys/time.h>

#if FTN_UNDERSCORE
#define DGEMM dgemm_
#define DGEMM_BATCH dgemm_batch_
#define DAXPY daxpy_
#else
#define DDOT ddot
#define DGEMM dgemm
#define DGEMM_BATCH dgemm_batch
#define DAXPY daxpy
#endif

static double __timer(){
  static bool initialized = false;
  static struct timeval start;
  struct timeval end;
  if(!initialized){
    gettimeofday( &start, NULL );
    initialized = true;
  }
  gettimeofday( &end, NULL );

  return (end.tv_sec - start.tv_sec) + 1.0e-6 * (end.tv_usec - start.tv_usec);
}

extern "C"
void DGEMM_BATCH(
          const char *,
          const char *,
          const int *,
          const int *,
          const int *,
          const double *,
          double **,
          const int *,
          double **,
          const int *,
          const double *,
          double **,
          const int *,
          const int *,
          const int *);



extern "C"
void DGEMM(const char *,
           const char *,
           const int *,
           const int *,
           const int *,
           const double *,
           const double *,
           const int *,
           const double *,
           const int *,
           const double *,
           double *,
           const int *);


extern "C"
void DAXPY(const int *    n,
           double *       dA,
           const double * dX,
           const int *    incX,
           double *       dY,
           const int *    incY);

void gemm(int m, int n, int k, double alpha, double const * A, double const * B, double beta, double * C){
  char cN = 'N';
  double alpha = 1.0;
  double beta = 0.0;
  DGEMM(&cN, &cN, &m, &n, &k, &alpha, A, &m, B, &k, &beta, &C, &m);
}

void axpy(int n, double alpha, double const * X, double * Y){
  int one = 1;
  DAXPY(&n, &alpha, X, &one, Y, &one);
}

double * naive_gemm(double const * A, double const * B, int64_t n, int64_t b){
  double * C = new double[n*b*n*b];
  int m = n*b;
  assert(n*b == (int64_t)m);
  //DGEMM(&cN, &cN, &m, &m, &m, &alpha, A, &m, B, &m, C, &m);
  gemm(m,m,m,alpha,A,B,C);
  //int mm = m*m;
  //assert(mm == ((int64_t)m)*((int64_t)m));
  double * C2 = new double[n*b*n*b];
  memcpy(C2, C, n*b*n*b*sizeof(double));
  int one = 1;
  for (int64_t i=0; i<m; i++){
    DAXPY(&m, &alpha, C2+i, &m, C+i*m, &one);
  }
  delete [] C2;
  return C;
}

double * fast_gemm(double const * pA, double const * pB, int64_t n, int64_t b){
  double * A = new double*[n*n];
  double * B = new double*[n*n];
  double * C = new double*[n*n];
  for (int64_t i=0; i<n; i++){
    for (int64_t j=0; j<=i; j++){
      A[i,j] = new double[b*b];
      A[j,i] = A[i,j];
      B[i,j] = new double[b*b];
      B[j,i] = B[i,j];
      for (int64_t ib=0; ib<b; ib++){
        for (int64_t jb=0; jb<b; jb++){
          A[i,j][jb+ib*b] = pA[(i*b+ib)*(n*b) + (j*b+jb)];
          B[i,j][jb+ib*b] = pB[(i*b+ib)*(n*b) + (j*b+jb)];
        }
      }
      C[i,j] = new double[b*b];
      C[j,i] = C[i,j];
      std::fill(C[i,j], C[i,j]+b*b, 0.);
    }
  }
  double * sA = new double*[n];
  double * sB = new double*[n];
  double * sC = new double*[n];
  for (int64_t i=0; i<n; i++){
    sA[i] = new double[b*b];
    std::fill(sA[i], sA[i]+b*b, 0.);
    for (int64_t j=0; j<n; j++){
      if (i==j)
        axpy(b*b,  1., A[i,j], sA[i]) 
      else 
        axpy(b*b, -1., A[i,j], sA[i]) 
    }
    sB[i] = new double[b*b];
    std::fill(sB[i], sB[i]+b*b, 0.);
    for (int64_t j=0; j<n; j++){
      if (i==j)
        axpy(b*b,  1., B[i,j], sB[i]) 
      else 
        axpy(b*b, -1., B[i,j], sB[i]) 
    }
    sC[i] = new double[b*b];
    std::fill(sC[i], sC[i]+b*b, 0.);
  }
  double * tmp_A = new double[b*b];
  double * tmp_B = new double[b*b];
  double * tmp_C = new double[b*b];
  for (int64_t i=0; i<n; i++){
    for (int64_t j=0; j<i; j++){
      for (int64_t k=0; k<j; k++){
        std::fill(tmp_A, tmp_A+b*b, 0.);
        std::fill(tmp_B, tmp_B+b*b, 0.);
        std::fill(tmp_C, tmp_C+b*b, 0.);
        axpy(b*b, 1., A[i,j], tmp_A);
        axpy(b*b, 1., A[i,k], tmp_A);
        axpy(b*b, 1., A[j,k], tmp_A);
        axpy(b*b, 1., B[i,j], tmp_B);
        axpy(b*b, 1., B[i,k], tmp_B);
        axpy(b*b, 1., B[j,k], tmp_B);
        gemm(b,b,b,1.0,tmp_A,tmp_B,1.0,tmp_C);
        axpy(b*b, 1., tmp_C, C[i,j]);
        axpy(b*b, 1., tmp_C, C[i,k]);
        axpy(b*b, 1., tmp_C, C[j,k]);
      }
    }
  }
  for (int64_t i=0; i<n; i++){
    for (int64_t j=0; j<i; j++){
      gemm(b,b,b,1.0,A[i,j],sB[i],1.0,C[i,j]);
      gemm(b,b,b,1.0,sA[i],B[i,j],1.0,C[i,j]);
      std::fill(tmp_C, tmp_C+b*b, 0.);
      gemm(b,b,b,1.0,A[i,j],B[i,j],1.0,tmp_C);
      axpy(b*b,  -(n-8.), tmp_C, C[i,j]) 
      axpy(b*b,  1., tmp_C, sC[i]) 
      axpy(b*b,  1., tmp_C, sC[j])
    }
  } 
  for (int64_t i=0; i<n; i++){
    for (int64_t j=0; j<i; j++){
      axpy(b*b, -1., sC[i], C[i,j]) 
      axpy(b*b, -1., sC[j], C[i,j]) 
    }
    axpy(b*b, 2., sC[i], C[i,i]) 
    gemm(b,b,b,2.0,A[i,i],B[i,i],1.0,C[i,i]);
  }
  double * pC = new double[n*b*n*b];
  for (int64_t i=0; i<n; i++){
    for (int64_t j=0; j<=i; j++){
      for (int64_t ib=0; ib<b; ib++){
        for (int64_t jb=0; jb<b; jb++){
          pC[(i*b+ib)*(n*b) + (j*b+jb)] = C[i,j][jb+ib*b];
          if (j!=i)
            pC[(j*b+jb)*(n*b) + (i*b+ib)] = C[i,j][jb+ib*b];
        }
      }
      delete A[i,j]
      delete B[i,j]
      delete C[i,j]
    }
    delete sA[i];
    delete sB[i];
    delete sC[i];
  }
  delete tmp_A;
  delete tmp_B;
  delete tmp_C;
  delete sA;
  delete sB;
  delete sC;
  delete A;
  delete B;
  delete C;

  for (int64_t i=0; i<n; i++){
    sC[i] = new double[b*b];
    std::fill(sC[i], sA[i]+b*b, 0.);
    for (int64_t j=0; j<n; j++){
      if (i==j)
        axpy(b*b,  1., A[i,j], sA[i]) 
      else 
        axpy(b*b, -1., A[i,j], sA[i]) 
    }
  }

  return C;
}


char* getCmdOption(char ** begin,
                   char ** end,
                   const   std::string & option){
  char ** itr = std::find(begin, end, option);
  if (itr != end && ++itr != end){
    return *itr;
  }
  return 0;
}

void bench_s1t1v1_blk(int64_t n, int64_t b, int niter){
  srand48(42);
  double * A = new double[n*b*n*b];
  double * B = new double[n*b*n*b];
  for (int i=0; i<n; i++){
    for (int j=0; j<=i; j++){
      for (int k=0; k<b; k++){
        for (int l=0; l<b; l++){
          A[(i*b+k)*n*b+(j*b+l)] = drand48();
          A[(j*b+k)*n*b+(i*b+l)] = A[(i*b+k)*n*b+(j*b+l)];
          B[(i*b+k)*n*b+(j*b+l)] = drand48();
          B[(j*b+k)*n*b+(i*b+l)] = B[(i*b+k)*n*b+(j*b+l)];
        }
      }
    }
  }
  printf("Naive times:\n");
  for (int i=0; i<niter; i++){
    double t_st = __time();
    double * C = naive_gemm(A,B,n,b);
    double t_end = __time();
    printf("%lf\n",t_end-t_st);
    delete [] C;
  }
  printf("Fast times:\n");
  for (int i=0; i<niter; i++){
    double t_st = __time();
    double * C = fast_gemm(A,B,n,b);
    double t_end = __time();
    printf("%lf\n",t_end-t_st);
    delete [] C;
  }
}

void test_s1t1v1_blk(int64_t n, int64_t b){
  srand48(42);
  double * A = new double[n*b*n*b];
  double * B = new double[n*b*n*b];
  for (int i=0; i<n; i++){
    for (int j=0; j<=i; j++){
      for (int k=0; k<b; k++){
        for (int l=0; l<b; l++){
          A[(i*b+k)*n*b+(j*b+l)] = drand48();
          A[(j*b+k)*n*b+(i*b+l)] = A[(i*b+k)*n*b+(j*b+l)];
          B[(i*b+k)*n*b+(j*b+l)] = drand48();
          B[(j*b+k)*n*b+(i*b+l)] = B[(i*b+k)*n*b+(j*b+l)];
        }
      }
    }
  }
  double * C_ref = naive_gemm(A,B,n,b);
  double * C_fst = fast_gemm(A,B,n,b);
  double err = 0.;
  double nrm = 0.;
  for (int i=0; i<n; i++){
    for (int j=0; j<=i; j++){
      for (int k=0; k<b; k++){
        for (int l=0; l<b; l++){
          double e = C_ref[(i*b+k)*n*b+(j*b+l)]-C_fst[(i*b+k)*n*b+(j*b+l)];
          err += e*e;
          double r = C_ref[(i*b+k)*n*b+(j*b+l)];
          nrm += r*r;
        }
      }
    }
  }
  printf("Relative Frobenius norm error between naive and fast is %lf\n",std::sqrt(err/nrm));
}

int main(int argc, char ** argv){
  int const in_num = argc;
  char ** input_str = argv;

  int64_t n, b;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;
  if (getCmdOption(input_str, input_str+in_num, "-b")){
    b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
    if (b < 0) b = 3;
  } else b = 3;

  test_s1t1v1_blk(n,b);
  return 0;
}
