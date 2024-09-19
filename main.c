#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"

void OpenCLConvolution2D(Matrix *input0, Matrix *input1, Matrix *result)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_a, device_b, device_c;

    cl_int err;

    cl_device_id device_id;    // device ID
    cl_context context;        // context
    cl_command_queue queue;    // command queue
    cl_program program;        // program
    cl_kernel kernel;          // kernel

    // Find platforms and devices
    OclPlatformProp *platforms = NULL;
    cl_uint num_platforms;

    err = OclFindPlatforms((const OclPlatformProp **)&platforms, &num_platforms);
    CHECK_ERR(err, "OclFindPlatforms");

    // Get ID for first device on first platform
    device_id = platforms[0].devices[0].device_id;

    // Create a context
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
    CHECK_ERR(err, "clCreateContext");

    // Create a command queue
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
    CHECK_ERR(err, "clCreateCommandQueueWithProperties");

    // Create the program from the source buffer
    program = clCreateProgramWithSource(context, 1, (const char **)&kernel_source, NULL, &err);
    CHECK_ERR(err, "clCreateProgramWithSource");

    // Build the program executable
    err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    CHECK_ERR(err, "clBuildProgram");

    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "convolution2D", &err);
    CHECK_ERR(err, "clCreateKernel");

    size_t size_a = sizeof(float) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS;
    size_t size_b = sizeof(float) * input1->shape[0] * input1->shape[1];
    size_t size_c = sizeof(float) * input0->shape[0] * input0->shape[1] * IMAGE_CHANNELS ;

    device_a = clCreateBuffer(context,CL_MEM_READ_ONLY , size_a, NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device a");
    device_b = clCreateBuffer(context,CL_MEM_READ_ONLY, size_b, NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device b");
    device_c = clCreateBuffer(context,CL_MEM_WRITE_ONLY, size_c, NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device c");

    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE ,0, size_a, input0->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device a");
    err = clEnqueueWriteBuffer(queue, device_b, CL_TRUE,0, size_b, input1->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device b");

    // Set the arguments to our compute kernel
    // __global float * inputData, __global float * outputData, __constant float * maskData,
    // int width, int height, int maskWidth,  int imageChannels
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &device_b);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &input0->shape[1]);
    CHECK_ERR(err, "clSetKernelArg 3");
    err |= clSetKernelArg(kernel, 4, sizeof(unsigned int), &input0->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 4");
    err |= clSetKernelArg(kernel, 5, sizeof(unsigned int), &input1->shape[0]);
    CHECK_ERR(err, "clSetKernelArg 5");
    int imageChannels = IMAGE_CHANNELS;
    err |= clSetKernelArg(kernel, 6, sizeof(unsigned int), &imageChannels);
    CHECK_ERR(err, "clSetKernelArg 6");

    // @@ define local and global work sizes
    size_t global_item_size[2] = {input0->shape[0], input0->shape[1]};
    size_t local_item_size[2] = {1,1};
    //@@ Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_item_size, local_item_size,0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE ,0, size_c, result->data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device c");
    //@@ Free the GPU memory here
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_b);
    clReleaseMemObject(device_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel); 
    clReleaseCommandQueue(queue); 
    clReleaseContext(context);
}

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
        return -1;
    }

    const char *input_file_a = argv[1];
    const char *input_file_b = argv[2];
    const char *input_file_c = argv[3];
    const char *input_file_d = argv[4];

    // Host input and output vectors and sizes
    Matrix host_a, host_b, host_c, answer;
    
    cl_int err;

    err = LoadImg(input_file_a, &host_a);
    CHECK_ERR(err, "LoadImg");

    err = LoadMatrix(input_file_b, &host_b);
    CHECK_ERR(err, "LoadMatrix");

    err = LoadImg(input_file_c, &answer);
    CHECK_ERR(err, "LoadImg");

    int rows, cols;
    //@@ Update these values for the output rows and cols of the output
    //@@ Do not use the results from the answer image
    rows = host_a.shape[0];
    cols = host_a.shape[1] 
    // Allocate the memory for the target.
    host_c.shape[0] = host_a.shape[0];
    host_c.shape[1] = host_a.shape[1]; 
    host_c.data = (float *)malloc(sizeof(float) * host_c.shape[0] * host_c.shape[1] * IMAGE_CHANNELS);

    OpenCLConvolution2D(&host_a, &host_b, &host_c);

    // Save the image
    SaveImg(input_file_d, &host_c);

    // Check the result of the convolution
    CheckImg(&answer, &host_c);

    // Release host memory
    free(host_a.data);
    free(host_b.data);
    free(host_c.data);
    free(answer.data);

    return 0;
}