CC=nvcc
CFLAGS=-w --ptxas-options=-v -O
#CFLAGS=-w --ptxas-options=-v -DPRINT_BOARDS -O
APPEXEC=gol
SOURCES=main.cu

all: $(APPEXEC)

$(APPEXEC): $(SOURCES) Makefile
	$(CC) $(CFLAGS) $(SOURCES) -o $(APPEXEC)

clean:
	rm -f $(APPEXEC)

