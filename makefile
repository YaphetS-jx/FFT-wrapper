all: test

CC = icc
CFLAGS = -g -DMKL_ILP64 -mkl=sequential -std=gnu99 -O3 -fopenmp
NVCC = nvcc
CUFLAGS = -lcudart -lcublas -lcufft


OBJSC = main.o Lap_Kron.o fft.o Lap_Matrix.o tools.o test.o Lap_Kron_GPU.o fft_GPU.o 

test: $(OBJSC)
	${CC} ${CFLAGS} $(CUFLAGS)  -o $@ $^ 

%.o: %.c %.h
	${CC} ${CFLAGS} -c $<

%.o: %.cu
	$(NVCC) $(CUFLAGS) -c $<

.PHONY: clean
clean:
	rm -f ./*.o *~ core core* test
