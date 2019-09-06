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

double * naive_gemm(double const * A, double const * B, int64_t n, int64_t b){
  double * C = new double[n*b*n*b];
  int m = n*b;
  assert(n*b == (int64_t)m);
  char cN = 'N';
  double alpha = 1.0;
  double beta = 0.0;
  DGEMM(&cN, &cN, &m, &m, &m, &alpha, A, &m, B, &m, C, &m);
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

double * fast_gemm(double const * A, double const * B, int64_t n, int64_t b){
  double * C = new double[n*b*n*b];
  int m = n*b;
  assert(n*b == (int64_t)m);
  char cN = 'N';
  double alpha = 1.0;
  double beta = 0.0;
  DGEMM(&cN, &cN, &m, &m, &m, &alpha, A, &m, B, &m, C, &m);
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


int main(int argc, char ** argv){
  int64_t n = 13, b = 4;

  MPI_Init(&argc, &argv);

  test_s1t1v1_blk(n,b);

  test_sq4d(n,b);

  MPI_Finalize();
  return 0;
}
