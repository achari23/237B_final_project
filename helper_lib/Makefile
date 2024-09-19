CC       = gcc
CFLAGS   = -g -Wall
INCFLAGS :=

ifeq ($(shell uname -o), Darwin)
	LDFLAGS = -framework OpenCL
else ifeq ($(shell uname -o), GNU/Linux) # Assumes NVIDIA GPU
	LDFLAGS  = -L/usr/local/cuda/lib64 -lOpenCL
	INCFLAGS += -I/usr/local/cuda/include
else # Android
	LDFLAGS = -lOpenCL
endif
LDFLAGS += -lm

SOURCES := device.c kernel.c matrix.c img.c
OBJECTS = $(SOURCES:.c=.o)

.PHONY: all
.SUFFIXES: .o .c
all: helper_lib.a

.c.o:
	$(CC) $(CFLAGS) -c -o $@ $^ $(INCFLAGS) $(LDFLAGS)

helper_lib.a: $(OBJECTS)
	ar rcs -o $@ $(OBJECTS)

clean: 
	rm -r $(OBJECTS) helper_lib.a