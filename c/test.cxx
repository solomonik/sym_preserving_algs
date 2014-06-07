#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <math.h>
#include <assert.h>
#include <vector>
#include <algorithm>
#include <ctf.hpp>

int factorial(int n){
  int f = 1;
  for (int i=2; i<=n; i++){
    f*=i;
  }
  return f;
}

int choose(int n, int c){
  return factorial(n)/factorial(c)/factorial(n-c);
}

int test(int n){
  int rank, num_pes;
  
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &num_pes);
  
  CTF_World dw(MPI_COMM_WORLD);

  int nn = n*n;
  int nns = n*(n+1)/2;

  CTF_Matrix A(nn,nn,NS,dw);
  CTF_Matrix B(nn,nn,NS,dw);
  CTF_Matrix C(nn,nn,NS,dw);
  
  int slid[] = {n,n,nns};
  int nlid[] = {n,n,nn};
  int syid[] = {SY,NS,NS};
  int asid[] = {AS,NS,NS};
  int nsid[] = {NS,NS,NS};
  
  CTF_Tensor syID(3, slid, syid, dw);
  CTF_Tensor asID(3, slid, asid, dw);
  CTF_Tensor nsID(3, nlid, nsid, dw);

  if (rank == 0){
    int64_t indices[nns];
    double vals[nns];
    int in = 0;
    for (int i=0; i<n; i++){
      for (int j=0; j<=i; j++){
        indices[in]=in*nns+i*n+j;
        vals[in]=1.0;
        in++;
      }
    }
    assert(nns==in);
    syID.write(nn,indices,vals);
  } else syID.write(0,NULL,NULL);

  if (rank == 0){
    int64_t indices[nns];
    double vals[nns];
    int in = 0;
    for (int i=0; i<n; i++){
      for (int j=0; j<i; j++){
        indices[in]=in*nns+i*n+j;
        vals[in]=1.0;
        in++;
      }
    }
    asID.write(in,indices,vals);
  } else asID.write(0,NULL,NULL);

  if (rank == 0){
    int64_t indices[nn];
    double vals[nn];
    int in = 0;
    for (int i=0; i<n; i++){
      for (int j=0; j<n; j++){
        indices[in]=in*nn+i*n+j;
        vals[in]=1.0;
        in++;
      }
    }
    assert(in==nn);
    nsID.write(in,indices,vals);
  } else nsID.write(0,NULL,NULL);

  int lens4[] = {n,n,n,n};
  int lens3[] = {n,n,nn};

  int sysy[] = {SY,NS,SY,NS};
  int syns[] = {SY,NS,NS};
  int asas[] = {AS,NS,AS,NS};
  int asns[] = {AS,NS,NS};

  CTF_Tensor syA(4, lens, sysy, dw);
  CTF_Tensor asA(4, lens, asas, dw);
  CTF_Tensor syB(3, lens, syns, dw);
  CTF_Tensor asB(3, lens, asns, dw);
  CTF_Tensor syC(3, lens, syns, dw);
  CTF_Tensor asC(3, lens, asns, dw);
  
  {
    int64_t * indices;
    double * values;
    int64_t size;
    srand48(173*rank);

    A.read_local(&size, &indices, &values);
    for (int i=0; i<size; i++){
      values[i] = drand48();
    }
    A.write(size, indices, values);
    free(indices);
    free(values);
    B.read_local(&size, &indices, &values);
    for (int i=0; i<size; i++){
      values[i] = drand48();
    }
    B.write(size, indices, values);
    free(indices);
    free(values);
  }

  C["ij"]=A["ik"]*B["kj"];
  C["ij"]+=B["ik"]*A["kj"];

  syA["ijkl"]=syID["ijI"]*A["IJ"]*syID["Jkl"];
  asA["ijkl"]=asID["ijI"]*A["IJ"]*syID["Jkl"];
  
  syB["klJ"]=syID["klI"]*B["IJ"];
  asB["klJ"]=asID["klI"]*B["IJ"];

  syC["ijJ"]=.5*syA["ijkl"]*syB["klJ"];
  asC["ijJ"]=.5*syA["ijkl"]*syB["klJ"];

  assert(syC.norm2() > 1.E-6);
  assert(asC.norm2() > 1.E-6);

  syC["ijJ"]-=syID["ijI"]*C["IJ"];
  asC["ijJ"]-=asID["ijI"]*C["IJ"];
  
  assert(syC.norm2() < 1.E-6);
  assert(asC.norm2() < 1.E-6);

  return 1;
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
  int rank, np, n, nv;
  int const in_num = argc;
  char ** input_str = argv;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  if (getCmdOption(input_str, input_str+in_num, "-n")){
    n = atoi(getCmdOption(input_str, input_str+in_num, "-n"));
    if (n < 0) n = 3;
  } else n = 3;
 
  int pass = test(n);
  assert(pass);
  
  MPI_Finalize();
  return 0;
}

