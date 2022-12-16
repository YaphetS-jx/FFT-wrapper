all: test

CC = icc

CFLAGS = -g -DMKL_ILP64 -mkl=sequential -std=gnu99 -O3 -fopenmp

OBJSC = main.o Lap_Kron.o fft.o Lap_Matrix.o Lap.o

test: $(OBJSC)
	${CC} ${CFLAGS} ${LIBS} -o $@ $^ 

%.o: %.c %.h
	${CC} ${CFLAGS} -c $<

.PHONY: clean
clean:
	rm -f ./*.o *~ core core* test
