#include "ctf.hpp"
using namespace CTF;

void s1t1v1_st_blk(Tensor<> & A, Tensor<> & B, Tensor<> & C){
  int symA[5] = {NS, NS, NS, A.sym[2], A.sym[3]};
  int lenA[5] = {A.lens[0], A.lens[1], C.lens[2], A.lens[2], A.lens[3]};
  Tensor<> ZA(5, lenA, symA);
  int symB[5] = {NS, NS, NS, B.sym[2], B.sym[3]};
  int lenB[5] = {B.lens[0], B.lens[1], C.lens[3], B.lens[2], B.lens[3]};
  ZA["ikcab"] += A["ikab"];
  Tensor<> ZB(5, lenB, symB);
  ZB["kjcab"] += B["kjab"];
//  Tensor<> ZC(5, len, sym);
  C["ijab"] += ZA["ikabc"]*ZB["kjbca"];
//  C["ijab"] += ZC["ijcab"];
}

bool test_s1t1v1_st_blk(int64_t n, int64_t b){
  int lens[] = {b, b, n, n};
  printf("Testing block direct evaluation algorithm for SY=SY*SY with n=%ld, b=%ld... ", n, b);

  int sym_SY[] = {NS, NS, SY, NS};
  Tensor<> A(4, lens, sym_SY);
  Tensor<> B(4, lens, sym_SY);
  Tensor<> C(4, lens, sym_SY);
  Tensor<> C2(4, lens, sym_SY);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  srand48(rank*347+23);

  A.fill_random(-1.0, 1.0);
  B.fill_random(-1.0, 1.0);

  C["ijab"] += A["ikac"]*B["kjcb"];

  s1t1v1_st_blk(A, B, C2);

  C2["ijab"] -= C["ijab"];

  double norm_ans = C.norm2();
  double norm_dif = C2.norm2();
  bool pass = norm_ans>=0.0 & norm_dif<=1.E-8*n*b;

  if (rank == 0){
    if (pass) printf("direct evaluation algorithm test passed.\n");
    else      printf("direct evaluation algorithm test FAILED, norm of answer is = %lf, norm of delta is = %lf\n.", norm_ans, norm_dif);
  }
  return pass;

}

int main(int argc, char ** argv){
  int64_t n = 8, b = 4;

  MPI_Init(&argc, &argv);

  bool pass = test_s1t1v1_st_blk(n,b);
  assert(pass);

  MPI_Finalize();
  return 0;
}

