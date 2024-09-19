#pragma once

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#define CL_TARGET_OPENCL_VERSION 300 // Use OpenCL 3.0
#include <CL/cl.h>
#endif

typedef struct _Matrix
{
    float *data;
    unsigned int shape[2];
} Matrix;

cl_int LoadMatrix(const char *path, Matrix *matrix);
cl_int SaveMatrix(const char *path, Matrix *matrix);
cl_int CheckMatrix(Matrix *truth, Matrix *student);
void PrintMatrix(Matrix *matrix);