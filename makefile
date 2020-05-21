all: fft

CC = icc

CFLAGS = -g -DMKL_ILP64 -mkl=sequential 

OBJSC = main.o fft.o

fft: $(OBJSC)
	${CC} ${CFLAGS} ${LIBS} -o $@ $^ 

%.o: %.c %.h
	${CC} ${CFLAGS} -c $<

.PHONY: clean
clean:
	rm -f ./*.o *~ core core*
