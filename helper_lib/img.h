#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300 // Use OpenCL 3.0
#include <CL/cl.h>
#endif

#include "matrix.h"

#define IMAGE_CHANNELS 3

cl_int LoadImg(const char *path, Matrix* img);
cl_int SaveImg(const char *path, Matrix *matrix);
cl_int CheckImg(Matrix *truth, Matrix *student);