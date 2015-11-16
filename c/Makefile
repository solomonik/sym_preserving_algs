CTF_DIR	 = /home/edgar/work/ctf/debug
FLAGS		 = -g -O0 -openmp
LIBS		 =	-L$(CTF_DIR)/lib -lctf -llapack -lblas
INCLUDES = -I$(CTF_DIR)/include
CC			 = mpicxx -cxx=icpc -std=c++11 $(FLAGS) $(INCLUDES)

#all: fast_ccsdt_ctr test
all: s1t1v1_blk

fast_ccsdt_ctr: fast_ccsdt_ctr.cxx
	$(CC) fast_ccsdt_ctr.cxx -o fast_ccsdt_ctr $(LIBS)

test: test.cxx
	$(CC) test.cxx -o test $(LIBS)

s1t1v1_blk: s1t1v1_blk.cxx Makefile $(CTF_DIR)/lib/libctf.a
	$(CC) s1t1v1_blk.cxx -o s1t1v1_blk $(LIBS)

clean:
	rm -f s1t1v1_blk
