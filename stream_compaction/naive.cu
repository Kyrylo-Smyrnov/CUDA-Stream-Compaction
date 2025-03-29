#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int threadsPerBlock = 256;

        int* d_idata;
        int* d_odata;

        __global__ void kernInclusiveScan(int n, int depth, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int step = 1 << (depth - 1);
            
            if (index - step < 0) {
                odata[index] = idata[index];
            }
            else {
                odata[index] = idata[index] + idata[index - step];
            }
        }

        __global__ void kernShiftRight(int n, int* odata, const int* idata) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            if (index == 0) {
                odata[index] = 0;
            }
            else {
                odata[index] = idata[index - 1];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            cudaMalloc((void**)&d_idata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_idata failed.");
            cudaMalloc((void**)&d_odata, n * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_odata failed.");

            cudaMemcpy(d_idata, idata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy d_idata <- idata failed.");
            cudaMemcpy(d_odata, odata, sizeof(int) * n, cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy d_odata <- odata failed.");

            int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            timer().startGpuTimer();

            for (int d = 1; d <= ilog2ceil(n); ++d) {
                kernInclusiveScan <<<numBlocks, threadsPerBlock>>> (n, d, d_odata, d_idata);
                std::swap(d_odata, d_idata);
            }

            kernShiftRight <<<numBlocks, threadsPerBlock>>> (n, d_odata, d_idata);

            timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy odata <- d_odata failed.");
            cudaFree(d_idata);
            checkCUDAErrorFn("cudaFree d_idata failed.");
            cudaFree(d_odata);
            checkCUDAErrorFn("cudaFree d_odata failed.");
        }
    }
}