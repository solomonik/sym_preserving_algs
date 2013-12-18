CTF_DIR	 = /home/edgar/work/ctf
FLAGS		 = -g -O0 -fopenmp
LIBS		 =	-L$(CTF_DIR)/lib -lctf -llapack -lblas
INCLUDES = -I$(CTF_DIR)/include
CC			 = mpicxx $(FLAGS) $(INCLUDES)

all: fast_ccsdt_ctr

fast_ccsdt_ctr: fast_ccsdt_ctr.cxx
	$(CC) fast_ccsdt_ctr.cxx -o fast_ccsdt_ctr $(LIBS)
