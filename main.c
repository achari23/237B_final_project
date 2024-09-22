#include <stdio.h>
#include <stdlib.h>

#include "device.h"
#include "kernel.h"
#include "matrix.h"
#include "img.h"
#include<math.h>

#define CHECK_ERR(err, msg)                           \
    if (err != CL_SUCCESS)                            \
    {                                                 \
        fprintf(stderr, "%s failed: %d\n", msg, err); \
        exit(EXIT_FAILURE);                           \
    }

#define KERNEL_PATH "kernel.cl"
  
//inputs is an array of 3 dim vectors 
//num_vectors is the number of vectors per input
//num_inputs is the number of inputs
//result is a an array of 5x1 dimensional vectors, same number of inputs (e.g. if 5 inputs then 25,1 output)
void Atanasov_Cal(Matrix *inputs, Matrix *result, int num_inputs, int num_vectors)
{
    // Load external OpenCL kernel code
    char *kernel_source = OclLoadKernel(KERNEL_PATH); // Load kernel source

    // Device input and output buffers
    cl_mem device_a, device_c;

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
    kernel = clCreateKernel(program, "atanasov", &err);
    CHECK_ERR(err, "clCreateKernel");

    size_t input_size, output_size;
    input_size = 0; 
    output_size = 0;
    for (int i = 0; i < num_inputs; i++) {
        output_size += sizeof(float) * 5; 
        input_size += sizeof(float) * 3*num_vectors; 
    }

    
    //need to squash input data into 1d
    float* input_data = malloc(input_size);
    printf("%d\n", num_vectors);
    for (int i = 0; i < num_inputs; i++ ) {
        for (int j = 0; j < num_vectors*3; j++) {
            input_data[i*num_vectors*3 + j] = inputs[i].data[j]; 
        }
        //memcpy(&(input_data[i*num_vectors*3]), &((inputs[i]).data), 3*num_vectors*sizeof(float));
    }  
    for (int i = 0; i< 3; i++ ) {
        printf("%f\n", input_data[i]);
    }
    //malloc a squashed output too 
    float* output = malloc(output_size);

    device_a = clCreateBuffer(context,CL_MEM_READ_ONLY , input_size, NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device a");
    device_c = clCreateBuffer(context,CL_MEM_WRITE_ONLY, output_size, NULL, &err );
    CHECK_ERR(err, "clCreateBuffer device c");

    //@@ Copy memory to the GPU here
    err = clEnqueueWriteBuffer(queue, device_a, CL_TRUE ,0, input_size, input_data, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueWriteBuffer device a");


    // Set the arguments to our compute kernel
    //  __global float * inputData, __global float * outputData, __constant int num_vectors, __constant int num_inputs){
    // int width, int height, int maskWidth,  int imageChannels
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &device_a);
    CHECK_ERR(err, "clSetKernelArg 0");
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &device_c);
    CHECK_ERR(err, "clSetKernelArg 1");
    err |= clSetKernelArg(kernel, 2, sizeof(unsigned int), &num_vectors);
    CHECK_ERR(err, "clSetKernelArg 2");
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &num_inputs);
    CHECK_ERR(err, "clSetKernelArg 3");
    

    // @@ define local and global work sizes
    size_t global_item_size = num_inputs;
    size_t local_item_size = 1;
    //@@ Launch the GPU Kernel here
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global_item_size, &local_item_size,0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueNDRangeKernel");
    //@@ Copy the GPU memory back to the CPU here
    err = clEnqueueReadBuffer(queue, device_c, CL_TRUE ,0, output_size, output, 0, NULL, NULL);
    CHECK_ERR(err, "clEnqueueReadBuffer device c");
    for (int i = 0; i < num_inputs; i++ ) {
        //memcpy(&(result[i].data), &output[i*5], 5* sizeof(float));
        for (int j = 0; j < 5; j++) {
       //     result[i].data[j] = output[i*5+j];
        }
    }
    for (int i = 0; i < 10; i++){
        printf("output %d %f\n",i, output[i]);
    }
        
    //@@ Free the GPU memory here
    clReleaseMemObject(device_a);
    clReleaseMemObject(device_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel); 
    clReleaseCommandQueue(queue); 
    clReleaseContext(context);
    free(input_data);
    free(output);
}

int main(int argc, char *argv[])
{
    //if (argc != 5)
   // {
   //     fprintf(stderr, "Usage: %s <input_file_0> <input_file_1> <answer_file> <output_file>\n", argv[0]);
  //      return -1;
  //  }


    // Host input and output vectors and sizes
    Matrix inputs[2];
    Matrix outputs[2];
    
    cl_int err;

    err = LoadMatrix("Dataset/0/input.raw", &inputs[0]);
    CHECK_ERR(err, "LoadMatrix0");

    err = LoadMatrix("Dataset/1/input.raw", &inputs[1]);
    CHECK_ERR(err, "LoadMatrix1");

    err = LoadMatrix("Dataset/0/output.raw", &outputs[0]);
    CHECK_ERR(err, "LoadMatrixoutput0");

    err = LoadMatrix("Dataset/1/output.raw", &outputs[1]);
    CHECK_ERR(err, "LoadMatrixoutput1");


    Matrix answers[2]; 
   
    answers[0].shape[0] = 1; 
    answers[0].shape[1] = 5;
    answers[1].shape[0] = 1; 
    answers[1].shape[1] = 5;

    int num_inputs = 2; 
    int num_vectors = inputs[0].shape[0]; 
    Atanasov_Cal(inputs, answers, num_inputs, num_vectors);

    // Save the image
    //SaveMatrix("Dataset/0/studentOut.raw", &answers[0]);
    //SaveMatrix("Dataset/1/studentOut.raw", &answers[1]);

    // Check the result of the atansov calibration
    //CheckMatrix(&outputs[0], &answers[0]);
    //CheckMatrix(&outputs[1], &answers[1]);

    // Release host memory
    

    return 0;
}