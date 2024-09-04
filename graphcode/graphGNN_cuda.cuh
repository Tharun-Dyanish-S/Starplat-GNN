#include <cuda.h>

// Matrix multiplication kernel for 2 1-D row major arrays
// As parellel as possible and dimensions are not equal
template <typename T>
__global__ void matrixMulRowMajor(T *A, T *B, T *C, int N, int M, int K)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < N && col < K)
  {
    T sum = 0;
    for (int i = 0; i < M; i++)
    {
      sum += A[row * M + i] * B[i * K + col];
    }
    C[row * K + col] = sum;
  }
}


__global__ void GCN_aggregate_cuda(float *aggregatedFeatures, float *postActivatedFeatures, 
                                   int *neighbors, float *weights, int *y_true, int *nodeLabels, 
                                   int num_neighbors, int num_features, int node) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_features) {
        float agg_sum = 0.0f;
        for (int k = 0; k < num_neighbors; k++) {
            int neighbor = neighbors[k];
            if (nodeLabels[neighbor] == nodeLabels[node]) {
                agg_sum += postActivatedFeatures[neighbor * num_features + i] * weights[k];
            }
        }
        aggregatedFeatures[node * num_features + i] = agg_sum;
    }
}

__global__ void forward_prop(float *aggregatedFeatures, float *weights, float *bias, 
                                           float *preActivatedFeatures, float *postActivatedFeatures, 
                                           int num_features, int prev_num_features, int node) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < num_features) {
        float sum = 0.0f;
        for (int j = 0; j < prev_num_features; j++) {
            sum += aggregatedFeatures[node * prev_num_features + j] * weights[j * num_features + i];
        }
        preActivatedFeatures[node * num_features + i] = sum + bias[i];

        postActivatedFeatures[node * num_features + i] = relu(preActivatedFeatures[node * num_features + i]);
    }
}

void forwardPass_cuda(GNN &gnn, int node, int layerNumber, int aggtype) 
{
    if (layerNumber == 0) {
        return;
    }
    std::vector<layer> &layers = gnn.getLayers();
    graph &g = gnn.getGraph();

    // Aggregate the features from the previous layer using CUDA
    if (aggtype == 1) {
        GCN_aggregate(gnn, node, layerNumber);
    } else if (aggtype == 2) {
        GIN_aggregate(gnn, node, layerNumber);
    }

    int num_features = layers[layerNumber].num_features;
    int prev_num_features = layers[layerNumber - 1].num_features;

    // Allocate device memory
    float *d_aggregatedFeatures, *d_weights, *d_bias, *d_preActivatedFeatures, *d_postActivatedFeatures;

    cudaMalloc(&d_aggregatedFeatures, prev_num_features * sizeof(float));
    cudaMalloc(&d_weights, prev_num_features * num_features * sizeof(float));
    cudaMalloc(&d_bias, num_features * sizeof(float));
    cudaMalloc(&d_preActivatedFeatures, num_features * sizeof(float));
    cudaMalloc(&d_postActivatedFeatures, num_features * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_aggregatedFeatures, &layers[layerNumber].aggregatedFeatures[node][0], 
               prev_num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, &layers[layerNumber].weights[0][0], 
               prev_num_features * num_features * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, &layers[layerNumber].bias[0], num_features * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the kernel for linear transformation and activation
    int blockSize = 256;
    int numBlocks = (num_features + blockSize - 1) / blockSize;
    linearTransformAndActivate<<<numBlocks, blockSize>>>(d_aggregatedFeatures, d_weights, d_bias, 
                                                         d_preActivatedFeatures, d_postActivatedFeatures, 
                                                         num_features, prev_num_features, node);

    // Copy result back to host
    cudaMemcpy(&layers[layerNumber].preActivatedFeatures[node][0], d_preActivatedFeatures, 
               num_features * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&layers[layerNumber].postActivatedFeatures[node][0], d_postActivatedFeatures, 
               num_features * sizeof(float), cudaMemcpyDeviceToHost);

    // If this is the last layer, apply softmax
    if (layerNumber == layers.size() - 1) {
        softmax(layers[layerNumber].postActivatedFeatures[node], num_features, layers[layerNumber].postActivatedFeatures[node]);
    }

    // Free device memory
    cudaFree(d_aggregatedFeatures);
    cudaFree(d_weights);
    cudaFree(d_bias);
    cudaFree(d_preActivatedFeatures);
    cudaFree(d_postActivatedFeatures);
}