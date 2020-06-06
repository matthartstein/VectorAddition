/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <time.h>
#include "support.h"
#include "kernel.cu"

int main(int argc, char**argv) {

    Timer timer;
    cudaError_t cuda_ret;
    time_t t;


    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    unsigned int n;
    if(argc == 1) {
        n = 10000;
    } else if(argc == 2) {
        n = atoi(argv[1]);
    } else {
        printf("\n    Invalid input parameters!"
           "\n    Usage: ./vecadd               # Vector of size 10,000 is used"
           "\n    Usage: ./vecadd <m>           # Vector of size m is used"
           "\n");
        exit(0);
    }
    
    /* Intializes random number generator */
    srand((unsigned) time(&t));    
    

    float* A_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { A_h[i] = (rand()%100)/100.00; }

    float* B_h = (float*) malloc( sizeof(float)*n );
    for (unsigned int i=0; i < n; i++) { B_h[i] = (rand()%100)/100.00; }

    float* C_h = (float*) malloc( sizeof(float)*n );

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Vector size = %u\n", n);

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    float* A_d;
    float* B_d;
    float* C_d;
    float bytes = sizeof(float) * n;
    
    cudaMalloc(&A_d, bytes);
    cudaMalloc(&B_d, bytes);
    cudaMalloc(&C_d, bytes);
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    cuda_ret = cudaMalloc((void**) &A_d, sizeof(float)*n);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaMalloc A_d = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaMalloc A_d = Passed\n");
    }
    
    cuda_ret = cudaMalloc((void**) &B_d, sizeof(float)*n);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaMalloc B_d = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaMalloc B_d = Passed\n");
    }
    
    cuda_ret = cudaMalloc((void**) &C_d, sizeof(float)*n);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaMalloc C_d = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaMalloc C_d = Passed\n");
    }

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(A_d, A_h, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, bytes, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    cuda_ret = cudaMemcpy(A_d, A_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaMemcpy (A_d, A_h) = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaMemcpy (A_d, A_h) = Passed\n");
    }
    
    cuda_ret = cudaMemcpy(B_d, B_h, sizeof(float)*n, cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaMemcpy (B_d, B_h) = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaMemcpy (B_d, B_h) = Passed\n");
    }
    

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    float NUM_THREADS = 256.0;
    float NUM_BLOCKS = (float)ceil(n / NUM_THREADS);
    
    vecAddKernel<<<NUM_BLOCKS, NUM_THREADS>>>(A_d, B_d, C_d, n);

    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
	else {printf(" = Passed...");}
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Copy device variables from host ----------------------------------------

    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    //INSERT CODE HERE
    cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    cuda_ret = cudaMemcpy(C_h, C_d, bytes, cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaMemcpy (C_h, C_d) = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaMemcpy (C_h, C_d) = Passed\n");
    }

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(A_h, B_h, C_h, n);

    // Free memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    //INSERT CODE HERE
    cuda_ret = cudaFree(A_d);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaFree (A_d, A_d) = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaFree (A_d, A_d) = Passed\n");
    }
    
    cuda_ret = cudaFree(B_d);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaFree (B_d, B_d) = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaFree (B_d, B_d) = Passed\n");
    }
    
    cuda_ret = cudaFree(C_d);
    if(cuda_ret != cudaSuccess) {
        printf("Testing cudaFree (C_h, C_d) = Failed\n");
        exit(-1);
    }
    else {
        printf("Testing cudaFree (C_h, C_d) = Passed\n\n");
    }
    
    return 0;
}