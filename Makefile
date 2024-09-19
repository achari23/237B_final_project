CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS := -I../../helper_lib
LDFLAGS  := ../../helper_lib/helper_lib.a -lm

ifeq ($(shell uname -o), Darwin)
	LDFLAGS += -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  += -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS += -lOpenCL
endif

all: solution

solution: ../../helper_lib/helper_lib.a main.c
	$(CC) $(CFLAGS) -o $@ $^ $(INCFLAGS) $(LDFLAGS)

../../helper_lib/helper_lib.a: 
	cd ../../helper_lib; make

run: solution
	./solution Dataset/0/input0.ppm Dataset/0/input1.raw Dataset/0/output.ppm output.ppm
	./solution Dataset/1/input0.ppm Dataset/1/input1.raw Dataset/1/output.ppm output.ppm
	./solution Dataset/2/input0.ppm Dataset/2/input1.raw Dataset/2/output.ppm output.ppm
	./solution Dataset/3/input0.ppm Dataset/3/input1.raw Dataset/3/output.ppm output.ppm
	./solution Dataset/4/input0.ppm Dataset/4/input1.raw Dataset/4/output.ppm output.ppm
	./solution Dataset/5/input0.ppm Dataset/5/input1.raw Dataset/5/output.ppm output.ppm
	./solution Dataset/6/input0.ppm Dataset/6/input1.raw Dataset/6/output.ppm output.ppm
	./solution Dataset/7/input0.ppm Dataset/7/input1.raw Dataset/7/output.ppm output.ppm
	./solution Dataset/8/input0.ppm Dataset/8/input1.raw Dataset/8/output.ppm output.ppm
	./solution Dataset/9/input0.ppm Dataset/9/input1.raw Dataset/9/output.ppm output.ppm
	
clean: 
	rm -f solution
	cd ../../helper_lib; make clean