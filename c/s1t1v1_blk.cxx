#include "ctf.hpp"
using namespace CTF;

void s1t1v1_st_blk(Tensor<> & A, Tensor<> & B, Tensor<> & C){
  int n = A.lens[2];
  int symA[5] = {NS, NS, NS, A.sym[2], A.sym[3]};
  int lenA[5] = {A.lens[0], A.lens[1], n, n, n};
  Tensor<> ZA(5, lenA, symA);
  int symB[5] = {NS, NS, NS, B.sym[2], B.sym[3]};
  int lenB[5] = {B.lens[0], B.lens[1], n, n, n};
  Tensor<> ZB(5, lenB, symB);
  int symC[5] = {NS, NS, NS, C.sym[2], C.sym[3]};
  int lenC[5] = {C.lens[0], C.lens[1], n, n, n};
  Tensor<> ZC(5, lenC, symC);
  ZA["ikcab"] += A["ikab"];
  ZB["kjcab"] += B["kjab"];
  C["ijab"] += ZA["ikbac"]*ZB["kjacb"];
  //ZC["ijcab"] += ZA["ikabc"]*ZB["kjbca"];
  //C["ijab"] += ZC["ijcab"];
}


void s1t1v1_sh_sh_fs_blk(Tensor<> & A, Tensor<> & B, Tensor<> & C){
  int n = A.lens[2];
  int symA[5] = {NS, NS, A.sym[2], A.sym[2], A.sym[3]};
  int lenA[5] = {A.lens[0], A.lens[1], n, n, n};
  Tensor<> ZA(5, lenA, symA);
  int symB[5] = {NS, NS, B.sym[2], B.sym[2], B.sym[3]};
  int lenB[5] = {B.lens[0], B.lens[1], n, n, n};
  Tensor<> ZB(5, lenB, symB);
  int symC[5] = {NS, NS, C.sym[2], C.sym[2], C.sym[3]};
  int lenC[5] = {C.lens[0], C.lens[1], n, n, n};
  Tensor<> ZC(5, lenC, symC);
  ZA["ikcab"] += A["ikab"];
  ZB["kjcab"] += B["kjab"];
  ZC["ijabc"] += ZA["ikabc"]*ZB["kjabc"];
  C["ijab"] += ZC["ijabc"];
  C["ijab"] -= A["ikac"]*B["kjac"];
  C["ijab"] -= A["ikac"]*B["kjab"];
  C["ijab"] -= A["ikab"]*B["kjac"];
  C["ijab"] -= ((int64_t)n-8)*A["ikab"]*B["kjab"];
}


void s1t1v1_as_as_fs_blk(Tensor<> & A, Tensor<> & B, Tensor<> & C){
  int n = A.lens[2];
  int symA[5] = {NS, NS, A.sym[2], A.sym[2], A.sym[3]};
  int lenA[5] = {A.lens[0], A.lens[1], n, n, n};
  Tensor<> ZA(5, lenA, symA);
  int symB[5] = {NS, NS, B.sym[2], B.sym[2], B.sym[3]};
  int lenB[5] = {B.lens[0], B.lens[1], n, n, n};
  Tensor<> ZB(5, lenB, symB);
  int symC[5] = {NS, NS, C.sym[2], C.sym[2], C.sym[3]};
  int lenC[5] = {C.lens[0], C.lens[1], n, n, n};
  Tensor<> ZC(5, lenC, symC);
  ZA["ikabc"] += A["ikab"];
  ZB["kjabc"] += B["kjab"];
  ZA = Tensor<>(ZA, symC);
  ZB = Tensor<>(ZB, symC);
  ZC["ijabc"] += ZA["ikabc"]*ZB["kjabc"];
  C["ijab"] += ZC["ijabc"];
  int sym_SH[] = {NS, NS, SH, NS};
  Tensor<> SH_A(A, sym_SH);
  Tensor<> SH_B(B, sym_SH);
  C["ijab"] -= SH_A["ikac"]*SH_B["kjac"];
  C["ijab"] += A["ikac"]*B["kjab"];
  C["ijab"] += A["ikab"]*B["kjac"];
  C["ijab"] -= ((double)n)*SH_A["ikab"]*SH_B["kjab"];
}

void s1t1v1_sh_as_fs_blk(Tensor<> & A, Tensor<> & B, Tensor<> & C){
  int n = A.lens[2];
  int symA[5] = {NS, NS, A.sym[2], A.sym[2], A.sym[3]};
  int lenA[5] = {A.lens[0], A.lens[1], n, n, n};
  Tensor<> ZA(5, lenA, symA);
  int symB[5] = {NS, NS, B.sym[2], B.sym[2], B.sym[3]};
  int lenB[5] = {B.lens[0], B.lens[1], n, n, n};
  Tensor<> ZB(5, lenB, symB);
  int symC[5] = {NS, NS, C.sym[2], C.sym[2], C.sym[3]};
  int lenC[5] = {C.lens[0], C.lens[1], n, n, n};
  Tensor<> ZC(5, lenC, symC);

  int sym_SH4[] = {NS, NS, SH, NS};
  int sym_SH5[] = {NS, NS, SH, SH, NS};

  ZA["ikabc"] += A["ikab"];
  ZB["kjabc"] += B["kjab"];
  ZA = Tensor<>(ZA, sym_SH5);
  ZB = Tensor<>(ZB, sym_SH5);
  ZC = Tensor<>(ZC, sym_SH5);
  ZC["ijabc"] += ZA["ikabc"]*ZB["kjabc"];
  ZC = Tensor<>(ZC, symC);
  C["ijab"] -= ZC["ijabc"];

  C["ijab"] -= A["ikac"]*B["kjac"];
  C["ijab"] += A["ikac"]*B["kjab"];
  C["ijab"] -= A["ikab"]*B["kjac"];

  Tensor<> SH_A(A, sym_SH4);
  Tensor<> SH_B(B, sym_SH4);
  Tensor<> SH_C(C, sym_SH4);
  SH_C["ijab"] += ((double)n)*SH_A["ikab"]*SH_B["kjab"];
  A = Tensor<>(SH_A, A.sym);
  B = Tensor<>(SH_B, B.sym);
  C = Tensor<>(SH_C, C.sym);
}


bool test_s1t1v1_blk(int64_t n, int64_t b, int sA, int sB, int sC, void (*func)(Tensor<>&,Tensor<>&,Tensor<>&)){
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int lens[] = {b, b, n, n};

  int sym_A[] = {NS, NS, sA, NS};
  int sym_B[] = {NS, NS, sB, NS};
  int sym_C[] = {NS, NS, sC, NS};
  Tensor<> A(4, lens, sym_A);
  Tensor<> B(4, lens, sym_B);
  Tensor<> C(4, lens, sym_C);
  Tensor<> C2(4, lens, sym_C);

  srand48(rank*347+23);

  A.fill_random(1.0, 1.0);
  B.fill_random(1.0, 1.0);

  C["ijab"] += A["ikac"]*B["kjcb"];

  func(A, B, C2);

  C.print();
  C2.print();

  C2["ijab"] -= C["ijab"];

  double norm_ans = C.norm2();
  double norm_dif = C2.norm2();
  bool pass = norm_ans>=0.0 & norm_dif<=1.E-8*n*b;

  if (!pass && rank == 0) printf("norm of answer is = %lf, norm of delta is = %lf\n.", norm_ans, norm_dif);
  return pass;
}

void test_s1t1v1_blk(int64_t n, int64_t b){
  int rank;
  bool pass;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank == 0)
    printf("Testing SH=SH*SH with n=%ld, b=%ld...\n", n, b);

  pass = test_s1t1v1_blk(n, b, SH, SH, SH, &s1t1v1_st_blk);
  
  if (rank == 0){
    if (pass) printf("  direct evaluation algorithm test passed.\n");
    else      printf("  direct evaluation algorithm test FAILED.\n");
  }

  pass = test_s1t1v1_blk(n, b, SH, SH, SH, &s1t1v1_sh_sh_fs_blk);

  if (rank == 0){
    if (pass) printf("  symmetry preserving algorithm test passed.\n");
    else      printf("  symmetry preserving algorithm test FAILED.\n");
  }

  if (rank == 0)
    printf("Testing SH=AS*AS with n=%ld, b=%ld...\n", n, b);
  
  pass = test_s1t1v1_blk(n, b, AS, AS, SH, &s1t1v1_st_blk);
 
  if (rank == 0){
    if (pass) printf("  direct evaluation algorithm test passed.\n");
    else      printf("  direct evaluation algorithm test FAILED.\n");
  }

  pass = test_s1t1v1_blk(n, b, AS, AS, SH, &s1t1v1_as_as_fs_blk);

  if (rank == 0){
    if (pass) printf("  symmetry preserving algorithm test passed.\n");
    else      printf("  symmetry preserving algorithm test FAILED.\n");
  }

  if (rank == 0)
    printf("Testing AS=AS*SH with n=%ld, b=%ld...\n", n, b);

  pass = test_s1t1v1_blk(n, b, AS, SH, AS, &s1t1v1_st_blk);
  
  if (rank == 0){
    if (pass) printf("  direct evaluation algorithm test passed.\n");
    else      printf("  direct evaluation algorithm test FAILED.\n");
  }

  pass = test_s1t1v1_blk(n, b, AS, SH, AS, &s1t1v1_sh_as_fs_blk);

  if (rank == 0){
    if (pass) printf("  symmetry preserving algorithm test passed.\n");
    else      printf("  symmetry preserving algorithm test FAILED.\n");
  }


  if (rank == 0)
    printf("Testing AS=SH*AS with n=%ld, b=%ld...\n", n, b);

  pass = test_s1t1v1_blk(n, b, SH, AS, AS, &s1t1v1_st_blk);
  
  if (rank == 0){
    if (pass) printf("  direct evaluation algorithm test passed.\n");
    else      printf("  direct evaluation algorithm test FAILED.\n");
  }


  pass = test_s1t1v1_blk(n, b, SH, AS, AS, &s1t1v1_sh_as_fs_blk);

  if (rank == 0){
    if (pass) printf("  symmetry preserving algorithm test passed.\n");
    else      printf("  symmetry preserving algorithm test FAILED.\n");
  }




}

int main(int argc, char ** argv){
  int64_t n = 4, b = 1;

  MPI_Init(&argc, &argv);

  test_s1t1v1_blk(n,b);

  MPI_Finalize();
  return 0;
}

