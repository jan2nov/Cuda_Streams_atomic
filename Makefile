###############################################################
# CUDA_HOME are supposed to be on default position
# and set it in your PATH .bashrc
###############################################################
INC := -I/usr/local/cuda/include
LIB := -L/usr/local/cuda/lib64 -lcudart -lcufft -lcuda

# use this compilers
# g++ just because the file write
GCC = g++
NVCC = nvcc


###############################################################
# Basic flags for compilers, one for debug options
# fmad flags used for reason of floating point operation
###############################################################
NVCCFLAGS = -O3 -arch=sm_70 --ptxas-options=-v -Xcompiler -fopenmp -Xcompiler -Wextra -lineinfo

GCC_OPTS =-O3 -Wall -fopenmp -Wextra $(INC)

ANALYZE = Stream_atomic_test.exe


ifdef reglim
NVCCFLAGS += --maxrregcount=$(reglim)
endif

ifdef fastmath
NVCCFLAGS += --use_fast_math
endif

all: clean analyze

analyze: Stream_atomic_test.o GPU_stream_atomic_test.o Makefile
	$(NVCC) -o $(ANALYZE) Stream_atomic_test.o GPU_stream_atomic_test.o $(LIB) $(NVCCFLAGS) 

GPU_stream_atomic_test.o: utils_cuda.h
	$(NVCC) -c GPU_stream_atomic_test.cu $(NVCCFLAGS)

Stream_atomic_test.o: Stream_atomic_test.cpp
	$(GCC) -c Stream_atomic_test.cpp $(GCC_OPTS)

clean:	
	rm -f *.o *.~ $(ANALYZE)


