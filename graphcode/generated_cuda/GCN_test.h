// FOR BC: nvcc bc_dsl_v2.cu -arch=sm_60 -std=c++14 -rdc=true # HW must support CC 6.0+ Pascal or after
#ifndef GENCPP_GCN_TEST_H
#define GENCPP_GCN_TEST_H
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <cuda.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../graph.cpp"
#include "../libcuda.cuh"

#include <cooperative_groups.h>

env environment("cuda", "static", "GCN_test");
void GCN(graph& g,NA g,thrust::device_vector<int> neuronsPerHiddenLayer);



// CSR BEGIN




// CSR END
//LAUNCH CONFIG
// TIMER START
__device__ int num_epoch ;
__device__ bool x ;
__device__ int layerr ;
__device__ bool y ;
__global__ void GCN_kernel(int V, int E){ // BEGIN KER FUN via ADDKERNEL
  float num_nodes  = V;
  unsigned v = blockIdx.x * blockDim.x + threadIdx.x;
  if(v >= V) return;
  if (layerr == neuronsPerHiddenLayer.size();
     + 1){ // if filter begin 
    y = false;

  } // if filter end
  gnn.forwardPass(nod,layerr);

} // end KER FUNC

#endif
