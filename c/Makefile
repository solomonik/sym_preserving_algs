CTF_DIR	  = /home/edgar/work/ctf/debug
FLAGS		  = -g -O0 -fopenmp
BLAS_LIBS	=	 -llapack -lblas
CTF_LIBS  =	-L$(CTF_DIR)/lib -lctf -llapack -lblas
INCLUDES  = -I$(CTF_DIR)/include
MPICXX	  = mpicxx -std=c++11 $(FLAGS) $(INCLUDES)
CC			  = g++ -std=c++11 $(FLAGS)

#all: fast_ccsdt_ctr test s1t1v1_blk
all: s1t1v1_blk_omp

fast_ccsdt_ctr: fast_ccsdt_ctr.cxx
	$(MPICXX) fast_ccsdt_ctr.cxx -o fast_ccsdt_ctr $(CTF_LIBS)

test: test.cxx
	$(MPICXX) test.cxx -o test $(CTF_LIBS)

s1t1v1_blk: s1t1v1_blk.cxx Makefile $(CTF_DIR)/lib/libctf.a
	$(MPICXX) s1t1v1_blk.cxx -o s1t1v1_blk $(CTF_LIBS)

s1t1v1_blk_seq: s1t1v1_blk_seq.cxx Makefile
	$(CC) s1t1v1_blk_seq.cxx -o s1t1v1_blk_seq $(BLAS_LIBS)

s1t1v1_blk_omp: s1t1v1_blk_omp.cxx Makefile
	$(CC) s1t1v1_blk_omp.cxx -o s1t1v1_blk_omp $(BLAS_LIBS)


clean:
	rm -f s1t1v1_blk s1t1v1_blk_seq s1t1v1_blk_omp test fast_cccsdt_ctr
