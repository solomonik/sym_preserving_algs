#include <string.h>
#include <string>
#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>
#include <list>
#include <vector>
#include <complex>
#include <unistd.h>
#include <iostream>
#include <limits.h>
#include <random>
#include <omp.h>

#include <sys/time.h>

#define FTN_UNDERSCORE 1


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

void print_matrix(double const * M, int n, int m){
  int i,j;
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      printf("%+2.4lf ", M[i+j*n]);
    }
    printf("\n");
  }
}
void print_matrixd(double ** M, int n, int m){
  int i,j;
  for (i = 0; i < n; i++){
    for (j = 0; j < m; j++){
      printf("%+2.4lf ", *(M[i+j*n]));
    }
    printf("\n");
  }
}

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

void get_omp_ij_phases(int & ii, int & ip, int & jj, int & jp){
  int tid = omp_get_thread_num();
  int ntd = omp_get_num_threads();
  ip = 1;
  jp = ntd/ip;
  while (ip < jp || ip*jp < ntd){
    ip++;
    jp = ntd/ip;
  }
  ii = tid / jp;
  jj = tid % jp;
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
  DGEMM(&cN, &cN, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
}

void axpy(int n, double alpha, double const * X, double * Y){
  int one = 1;
  DAXPY(&n, &alpha, X, &one, Y, &one);
}

double * naive_gemm(double const * A, double const * B, int64_t n, int64_t b){
  double * C = new double[n*b*n*b];
  int m = n*b;
  assert(n*b == (int64_t)m);
  double t_st = __timer();
  gemm(m,m,m,1.,A,B,0.,C);
  double t_end = __timer();
  printf("gemm took %lf sec\n",t_end-t_st);
  //int mm = m*m;
  //assert(mm == ((int64_t)m)*((int64_t)m));
  //double * C2 = new double[n*b*n*b];
  //memcpy(C2, C, n*b*n*b*sizeof(double));
  //int one = 1;
  //for (int64_t i=0; i<m; i++){
  //  double alpha = 1.;
  //  DAXPY(&m, &alpha, C2+i, &m, C+i*m, &one);
  //}
  #pragma omp parallel
  {
    int ii, ip, jj, jp;
    get_omp_ij_phases(ii, ip, jj, jp);
    for (int64_t i=ii; i<n; i+=ip){
      for (int64_t j=jj; j<=i; j+=jp){
        for (int k=0; k<b; k++){
          for (int l=0; l<b; l++){
            C[(j*b+k)*n*b+(i*b+l)] += C[(i*b+k)*n*b+(j*b+l)];
            C[(i*b+k)*n*b+(j*b+l)]  = C[(j*b+k)*n*b+(i*b+l)];
          }
        }
      }
    }
  }

  //delete [] C2;
  return C;
}

double * fast_gemm(double const * pA, double const * pB, int64_t n, int64_t b){
  double ** A = new double*[n*n];
  double ** B = new double*[n*n];
  double ** C = new double*[n*n];
  #pragma omp parallel
  {
    int ii, ip, jj, jp;
    get_omp_ij_phases(ii, ip, jj, jp);
    for (int64_t i=ii; i<n; i+=ip){
      for (int64_t j=jj; j<=i; j+=jp){
        A[i*n+j] = new double[b*b];
        A[j*n+i] = A[i*n+j];
        B[i*n+j] = new double[b*b];
        B[j*n+i] = B[i*n+j];
        for (int64_t ib=0; ib<b; ib++){
          for (int64_t jb=0; jb<b; jb++){
            A[i*n+j][jb+ib*b] = pA[(i*b+ib)*(n*b) + (j*b+jb)];
            B[i*n+j][jb+ib*b] = pB[(i*b+ib)*(n*b) + (j*b+jb)];
          }
        }
        C[i*n+j] = new double[b*b];
        C[j*n+i] = C[i*n+j];
        std::fill(C[i*n+j], C[i*n+j]+b*b, 0.);
      }
    }
  }
  double ** sA = new double*[n];
  double ** sB = new double*[n];
  double ** sC = new double*[n];
  #pragma omp parallel for
  for (int64_t i=0; i<n; i++){
    sA[i] = new double[b*b];
    std::fill(sA[i], sA[i]+b*b, 0.);
    for (int64_t j=0; j<n; j++){
      if (i==j)
        axpy(b*b,  1., A[i*n+j], sA[i]);
      else 
        axpy(b*b, -1., A[i*n+j], sA[i]);
    }
    sB[i] = new double[b*b];
    std::fill(sB[i], sB[i]+b*b, 0.);
    for (int64_t j=0; j<n; j++){
      if (i==j)
        axpy(b*b,  1., B[i*n+j], sB[i]); 
      else 
        axpy(b*b, -1., B[i*n+j], sB[i]); 
    }
    sC[i] = new double[b*b];
    std::fill(sC[i], sC[i]+b*b, 0.);
  }
  omp_lock_t * locks = new omp_lock_t[n*n];
  for (int i=0; i<n*n; i++){
    omp_init_lock(locks+i);
  }
  double t_st = __timer();
  #pragma omp parallel
  {
    double * tmp_A = new double[b*b];
    double * tmp_B = new double[b*b];
    double * tmp_C = new double[b*b];
    int tid = omp_get_thread_num();
    int ntd = omp_get_num_threads();
    int s = 0;
    for (int64_t i=0; i<n; i++){
      for (int64_t j=0; j<i; j++){
        for (int64_t k=0; k<j; k++){
          if (s%ntd != tid){
            s++;
            continue;
          } else s++;
          std::fill(tmp_A, tmp_A+b*b, 0.);
          std::fill(tmp_B, tmp_B+b*b, 0.);
          std::fill(tmp_C, tmp_C+b*b, 0.);
          axpy(b*b, 1., A[i*n+j], tmp_A);
          axpy(b*b, 1., A[i*n+k], tmp_A);
          axpy(b*b, 1., A[j*n+k], tmp_A);
          axpy(b*b, 1., B[i*n+j], tmp_B);
          axpy(b*b, 1., B[i*n+k], tmp_B);
          axpy(b*b, 1., B[j*n+k], tmp_B);
          gemm(b,b,b,1.0,tmp_A,tmp_B,1.0,tmp_C);
          omp_set_lock(&locks[i*n+j]);
          axpy(b*b, 1., tmp_C, C[i*n+j]);
          omp_unset_lock(&locks[i*n+j]);
          omp_set_lock(&locks[i*n+k]);
          axpy(b*b, 1., tmp_C, C[i*n+k]);
          omp_unset_lock(&locks[i*n+k]);
          omp_set_lock(&locks[j*n+k]);
          axpy(b*b, 1., tmp_C, C[j*n+k]);
          omp_unset_lock(&locks[j*n+k]);
        }
      }
    }
    delete [] tmp_A;
    delete [] tmp_B;
    delete [] tmp_C;
  }
  double t_end = __timer();
  printf("main fast gemm loop took %lf sec\n",t_end-t_st);
  delete [] locks;
  locks = new omp_lock_t[n];
  for (int i=0; i<n; i++){
    omp_init_lock(locks+i);
  }
  #pragma omp parallel
  {
    double * tmp_C = new double[b*b];
    int ii, ip, jj, jp;
    get_omp_ij_phases(ii, ip, jj, jp);
    for (int64_t i=ii; i<n; i+=ip){
      for (int64_t j=jj; j<i; j+=jp){
        gemm(b,b,b,1.0,A[i*n+j],sB[i],1.0,C[i*n+j]);
        gemm(b,b,b,1.0,A[i*n+j],sB[j],1.0,C[i*n+j]);
        gemm(b,b,b,1.0,sA[i],B[i*n+j],1.0,C[i*n+j]);
        gemm(b,b,b,1.0,sA[j],B[i*n+j],1.0,C[i*n+j]);
        std::fill(tmp_C, tmp_C+b*b, 0.);
        gemm(b,b,b,1.0,A[i*n+j],B[i*n+j],1.0,tmp_C);
        axpy(b*b,  -(n-8.), tmp_C, C[i*n+j]); 
        omp_set_lock(&locks[i]);
        axpy(b*b,  1., tmp_C, sC[i]);
        omp_unset_lock(&locks[i]);
        omp_set_lock(&locks[j]);
        axpy(b*b,  1., tmp_C, sC[j]);
        omp_unset_lock(&locks[j]);
      }
    }
    delete [] tmp_C;
  }
  delete [] locks;
  #pragma omp parallel
  {
    int ii, ip, jj, jp;
    get_omp_ij_phases(ii, ip, jj, jp);
    for (int64_t i=ii; i<n; i+=ip){
      for (int64_t j=jj; j<i; j+=jp){
        axpy(b*b, -1., sC[i], C[i*n+j]);
        axpy(b*b, -1., sC[j], C[i*n+j]); 
      }
    }
  }
  #pragma omp parallel for
  for (int64_t i=0; i<n; i++){
    axpy(b*b, 2., sC[i], C[i*n+i]);
    gemm(b,b,b,2.0,A[i*n+i],B[i*n+i],1.0,C[i*n+i]);
  }

  double * pC = new double[n*b*n*b];
  #pragma omp parallel
  {
    int ii, ip, jj, jp;
    get_omp_ij_phases(ii, ip, jj, jp);
    for (int64_t i=ii; i<n; i+=ip){
      for (int64_t j=jj; j<=i; j+=jp){
        for (int64_t ib=0; ib<b; ib++){
          for (int64_t jb=0; jb<b; jb++){
            pC[(i*b+ib)*(n*b) + (j*b+jb)] = C[i*n+j][jb+ib*b];
            if (j!=i)
              pC[(j*b+ib)*(n*b) + (i*b+jb)] = C[i*n+j][jb+ib*b];
          }
        }
      }
    }
  }
  for (int64_t i=0; i<n; i++){
    for (int64_t j=0; j<=i; j++){
      delete [] A[i*n+j];
      delete [] B[i*n+j];
      delete [] C[i*n+j];
    }
    delete [] sA[i];
    delete [] sB[i];
    delete [] sC[i];
  }
  delete [] sA;
  delete [] sB;
  delete [] sC;
  delete [] A;
  delete [] B;
  delete [] C;

  return pC;
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

void bench_s1t1v1_blk(int64_t n, int64_t b, int niter_nv, int niter_fs){
  srand48(42);
  double * A = new double[n*b*n*b];
  double * B = new double[n*b*n*b];
  for (int i=0; i<n; i++){
    for (int j=0; j<=i; j++){
      for (int k=0; k<b; k++){
        for (int l=0; l<b; l++){
          A[(i*b+k)*n*b+(j*b+l)] = drand48()-.5;
          A[(j*b+k)*n*b+(i*b+l)] = A[(i*b+k)*n*b+(j*b+l)];
          B[(i*b+k)*n*b+(j*b+l)] = drand48()-.5;
          B[(j*b+k)*n*b+(i*b+l)] = B[(i*b+k)*n*b+(j*b+l)];
        }
      }
    }
  }
  std::vector<double> ntimes;
  if (niter_nv > 0){
    printf("Naive times:\n");
    double t_nv = 0.;
    for (int i=0; i<niter_nv; i++){
      double t_st = __timer();
      double * C = naive_gemm(A,B,n,b);
      double t_end = __timer();
      double t = t_end - t_st;
      ntimes.push_back(t);
      printf("%lf\n",t);
      t_nv += t;
      delete [] C;
    }
    printf("Average time for naive is \n");
    printf("%lf\n",t_nv/niter_nv);
    printf("Median time for naive is \n");
    std::sort(&ntimes[0], &ntimes[0]+ntimes.size());
    printf("%lf\n",ntimes[ntimes.size()/2]);
  }
  std::vector<double> ftimes;
  if (niter_fs > 0){
    printf("Fast times:\n");
    double t_fs = 0.;
    for (int i=0; i<niter_fs; i++){
      double t_st = __timer();
      double * C = fast_gemm(A,B,n,b);
      double t_end = __timer();
      double t = t_end - t_st;
      ftimes.push_back(t);
      printf("%lf\n",t);
      t_fs += t;
      delete [] C;
    }
    printf("Average time for naive is \n");
    printf("%lf\n",t_fs/niter_fs);
    printf("Median time for naive is \n");
    std::sort(&ftimes[0], &ftimes[0]+ftimes.size());
    printf("%lf\n",ftimes[ftimes.size()/2]);
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
          A[(i*b+k)*n*b+(j*b+l)] = drand48()-.5;
          A[(j*b+k)*n*b+(i*b+l)] = A[(i*b+k)*n*b+(j*b+l)];
          B[(i*b+k)*n*b+(j*b+l)] = drand48()-.5;
          B[(j*b+k)*n*b+(i*b+l)] = B[(i*b+k)*n*b+(j*b+l)];
        }
      }
    }
  }
  double * C_ref = naive_gemm(A,B,n,b);
  double * C_fst = fast_gemm(A,B,n,b);
  //printf("C_ref:\n");
  //print_matrix(C_ref,n*b,n*b);
  //printf("C_fst:\n");
  //print_matrix(C_fst,n*b,n*b);
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
  int test, niter_nv, niter_fs;
  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 4;
  } else n = 4;
  if (getCmdOption(input_str, input_str+in_num, "-b")){
    b = atoi(getCmdOption(input_str, input_str+in_num, "-b"));
    if (b < 0) b = 3;
  } else b = 3;
  if (getCmdOption(input_str, input_str+in_num, "-niter_nv")){
    niter_nv = atoi(getCmdOption(input_str, input_str+in_num, "-niter_nv"));
    if (niter_nv < 0) niter_nv = 3;
  } else niter_nv = 3;
  if (getCmdOption(input_str, input_str+in_num, "-niter_fs")){
    niter_fs = atoi(getCmdOption(input_str, input_str+in_num, "-niter_fs"));
    if (niter_fs < 0) niter_fs = 3;
  } else niter_fs = 3;
  if (getCmdOption(input_str, input_str+in_num, "-test")){
    test = atoi(getCmdOption(input_str, input_str+in_num, "-test"));
    if (test < 0) test = 1;
  } else test = 1;

  if (test > 0){
    printf("Testing n=%ld b=%ld\n",n,b);
    test_s1t1v1_blk(n,b);
  }
  printf("Benchmarking n=%ld b=%ld niter_nv=%d niter_fs=%d\n",n,b,niter_nv,niter_fs);
  bench_s1t1v1_blk(n,b,niter_nv,niter_fs);
  return 0;
}
