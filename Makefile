CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I helper_lib
LDFLAGS  := helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

solution: helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

helper_lib/helper_lib.a: 
	cd helper_lib; make

run: solution
	./solution 
	
clean: 
	rm -f solution
	cd helper_lib; make clean