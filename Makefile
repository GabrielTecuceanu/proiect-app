# make serial   -> build/knn_serial   (C99, doar gcc; fara g++)
# Variantele paralele: target nou cu acelasi main.c + src/knn_<var>.c
#   omp: -fopenmp ;  mpi: mpicc ;  cuda: nvcc
CC       ?= gcc
ARCH     ?= -march=native
CFLAGS   ?= -O3 $(ARCH) -std=gnu99 -Wall -Wextra
INC      := -Iinclude
COMMON   := src/main.c src/common.c

all: serial omp

build/knn_serial: $(COMMON) src/knn_serial.c $(wildcard include/*.h)
	@mkdir -p build
	$(CC) $(CFLAGS) $(INC) -DKNN_IMPL='"serial"' $(COMMON) src/knn_serial.c -o $@ -lm

build/knn_omp: $(COMMON) src/knn_omp.c $(wildcard include/*.h)
	@mkdir -p build
	$(CC) $(CFLAGS) -fopenmp $(INC) -DKNN_IMPL='"omp"' $(COMMON) src/knn_omp.c -o $@ -lm

serial: build/knn_serial
omp: build/knn_omp

clean:
	rm -rf build

.PHONY: all serial omp clean
