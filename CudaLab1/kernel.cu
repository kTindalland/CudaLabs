
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

int threadsPerBlock;
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b, int threadsPerBlock)
{
    int i = threadIdx.x + blockIdx.x * threadsPerBlock;
    c[i] = a[i] + b[i];
}

__global__ void fillArrayKernel(int* a, int num, int threadsPerBlock) {
    int i = threadIdx.x + blockIdx.x * threadsPerBlock;
    int result = num + i;
    a[i] = result;
}

__global__ void printKernel(int* c, int threadsPerBlock) {
    int i = threadIdx.x + blockIdx.x * threadsPerBlock;
    printf("%d\n", i);
}

int main()
{
    const int arraySize = 5000;
    const int a[arraySize] = { 1 };
    const int b[arraySize] = { 10 };
    int c[arraySize] = { 0 };
    threadsPerBlock = 1000;
    cudaError_t cudaStatus;
    cudaEvent_t start, stop;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;

    cudaStatus = cudaMalloc((void**)&dev_a, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_c, arraySize * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_a, a, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_c, c, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    fillArrayKernel << <5, threadsPerBlock >> > (dev_a, 10, threadsPerBlock);
    fillArrayKernel << <5, threadsPerBlock >> > (dev_b, 1, threadsPerBlock);

    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    // <<< Number_of_blocks, Number_of_threads / block >>>
    addKernel <<<5, threadsPerBlock>>> (dev_c, dev_a, dev_b, threadsPerBlock);
    cudaEventRecord(stop, 0);

    cudaStatus = cudaDeviceSynchronize(); // Makes sure all threads finished.

    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);


    printKernel<<<5, threadsPerBlock>>>(dev_c, threadsPerBlock);
    cudaDeviceSynchronize();

    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(c, dev_c, arraySize * sizeof(int), cudaMemcpyDeviceToHost); // Copy result back from the GPUcudaMemcpy
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    
    
    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    printf("Time elapsed the execution of Kernel: %f\n", elapsedTime);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
   /* cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }*/

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return 0;
}
