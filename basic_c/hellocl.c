/*
 *  This file started out as a sample I downloaded from either
 *  Intel, Apple, or some other source.
 *
 *  TODO: find the original and make sure attributions are made
 *        and all permissions are being correctly followed
 *
 *   Just found it:

https://www.olcf.ornl.gov/tutorials/opencl-vector-addition/#vecAdd.c

 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

 
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                      "\n" \
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
                                                                "\n" ;
/* const char *kernelSource = "                                      "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global float *a,                       \n" \
"                       __global float *b,                       \n" \
"                       __global float *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
 */
                                                                
int main( int argc, char* argv[] )
{
    // Length of vectors
    unsigned int n = 16*1024*1024;
    double index;

    // Host input vectors
    float *h_a;
    float *h_b;
    // Host output vector
    float *h_c;
 
    // Device input buffers
    cl_mem d_a;
    cl_mem d_b;
    // Device output buffer
    cl_mem d_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel

    setbuf(stdout, NULL);
    
    // Size, in bytes, of each vector
    size_t bytes = 2*n*sizeof(float);
 
    // Allocate memory for each vector on host
    printf("// Allocate memory for each vector on host\n");
    h_a = (float*)malloc(bytes);
    h_b = (float*)malloc(bytes);
    h_c = (float*)malloc(bytes);
 
    // Initialize vectors on host
    printf("// Initialize vectors on host\n");
    int i;
    for( i = 0; i < 2*n; i++ )
    {
        index = (double)(-n + i);
        h_a[i] = sin(index)*sin(index);
        h_b[i] = cos(index)*cos(index);
    }
    
    printf("// Done with memory setup\n");
    
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(2*n/(float)localSize)*localSize;
 
    // Bind to platform
    printf("// Bind to platform\n");
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    printf("// Get ID for the device\n");
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);

    // Create a context  
    printf("// Create a context  \n");
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);

    
    // Create a command queue 
    printf("// Create a command queue \n");
    queue = clCreateCommandQueue(context, device_id, 0, &err);

    // Create the compute program from the source buffer
    printf("// Create the compute program from the source buffer\n");
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
 
    // Build the program executable 
    printf("// Build the program executable \n");
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    printf("// Create the compute kernel in the program we wish to run\n");
    kernel = clCreateKernel(program, "vecAdd", &err);
 
    // Create the input and output arrays in device memory for our calculation
    printf("// Create the input and output arrays in device memory for our calculation\n");
    d_a = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_b = clCreateBuffer(context, CL_MEM_READ_ONLY, bytes, NULL, NULL);
    d_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bytes, NULL, NULL);
    
    // Write our data set into the input array in device memory
    printf("// Write our data set into the input array in device memory\n");
    err = clEnqueueWriteBuffer(queue, d_a, CL_TRUE, 0,
                                   bytes, h_a, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer(queue, d_b, CL_TRUE, 0,
                                   bytes, h_b, 0, NULL, NULL);
 
    // Set the arguments to our compute kernel
    printf("// Set the arguments to our compute kernel\n");
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_a);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_b);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_c);
    unsigned int num_elements = 2*n;
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &num_elements);
 
    // Execute the kernel over the entire range of the data set  
    printf("// Execute the kernel over the entire range of the data set  \n");
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    printf("// Wait for the command queue to get serviced before reading back results\n");
    clFinish(queue);

    // Read the results from the device
    printf("// Read the results from the device\n");
    clEnqueueReadBuffer(queue, d_c, CL_TRUE, 0,
                                bytes, h_c, 0, NULL, NULL );
 
    // Sum up vector c and print result divided by n, this should equal 1 within error
    printf("// Sum up vector c and print result divided by n,\n");
    printf("//   this should equal 1 within error\n");
    float sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);
 
    // release OpenCL resources
    printf("// release OpenCL resources\n");
    clReleaseMemObject(d_a);
    clReleaseMemObject(d_b);
    clReleaseMemObject(d_c);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    free(h_a);
    free(h_b);
    free(h_c);
 
    printf ("Done!");
    return 0;
}
