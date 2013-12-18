#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <ctf.hpp>

int fast_ccsdt_ctr(int no, int nv, CTF_World &dw){
  int rank, i, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);

  int lenW[] = {no,nv,no,nv};
  int lenT[] = {no,no,no,nv,nv,nv};
  int symW[] = {NS,NS,NS,NS};
  int symT[] = {AS,AS,NS,AS,AS,NS};

  CTF_Tensor W(4, lenW, symW, dw);
  CTF_Tensor T(6, lenT, symT, dw);
  CTF_Tensor Z(6, lenT, symT, dw);
  CTF_Tensor Z_ans(6, lenT, symT, dw);

  {
    int64_t * indices;
    double * values;
    int64_t size;
    srand48(173*rank);

    W.read_local(&size, &indices, &values);
    for (i=0; i<size; i++){
      values[i] = drand48();
    }
    W.write(size, indices, values);
    free(indices);
    free(values);
    T.read_local(&size, &indices, &values);
    for (i=0; i<size; i++){
      values[i] = drand48();
    }
    T.write(size, indices, values);
    free(indices);
    free(values);
  }
  Z_ans["abcijk"] += W["amei"]*T["ebcmjk"];

  assert(Z_ans.norm2() > 1.E-9);

  int lenW_rep[] = {no,no,no,nv,no,nv};
  int symW_rep[] = {SH,SH,NS,NS,NS,NS};
  CTF_Tensor W_rep(6, lenW_rep, symW_rep, dw);
  CTF_Tensor W_sub(5, lenW_rep+1, symW_rep+1, dw);
  
  W_rep["abcmei"] += W["amei"];
  W_sub["bcmei"] += W_rep["abcmei"];
  W_sub["bcmei"] += ((double)no)*W["bmei"];
  Z["abcijk"] += W_rep["abcmei"]*T["ebcmjk"];
  Z["abcijk"] -= W_sub["bcmei"]*T["ebcmjk"];
  
  Z_ans.print();
  Z.print();

  Z["abcijk"] -= Z_ans["abcijk"];


  return (Z.norm2() < 1.E-9);
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


int main(int argc, char ** argv){
  int rank, np, no, nv;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-no")){
    no = atoi(getCmdOption(input_str, input_str+in_num, "-no"));
    if (no < 0) no = 3;
  } else no = 3;
  if (getCmdOption(input_str, input_str+in_num, "-nv")){
    nv = atoi(getCmdOption(input_str, input_str+in_num, "-nv"));
    if (nv < 0) nv = 3;
  } else nv = 3;

  {
    CTF_World dw(MPI_COMM_WORLD, argc, argv);
    if (rank == 0){
      printf("Computing Z^(abc)_(ijk) = W_{am}^{ei} * T_(ebc)(mjk)\n");
    }
    int pass = fast_ccsdt_ctr(no, nv, dw);
    assert(pass);
  }
  
  MPI_Finalize();
  return 0;
}

