#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        const int threadsPerBlock = 256;

        int* d_idata;
        int* d_odata;
        int* d_bools;
        int* d_indices;

        __global__ void kernUpSweep(int n, int depth, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int step = 1 << depth;
            int sumOffset = 1 << (depth + 1);

            int i = index * sumOffset + sumOffset - 1;
            data[i] += data[i - step];
        }

        __global__ void kernDownSweep(int n, int depth, int* data) {
            int index = threadIdx.x + (blockIdx.x * blockDim.x);
            if (index >= n) {
                return;
            }

            int step = 1 << (depth + 1);
            int halfStep = 1 << depth;

            int right = index * step + step - 1;
            int left = index * step + halfStep - 1;

            int tmp = data[left];
            data[left] = data[right];
            data[right] += tmp;
        }

        void scanWithoutTimer(int n, int *data) {
            for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
                int activeThreads = n >> (d + 1);
                int numBlocks = (activeThreads + threadsPerBlock - 1) / threadsPerBlock;
                kernUpSweep <<<numBlocks, threadsPerBlock>>> (activeThreads, d, data);
            }

            cudaMemset(data + n - 1, 0, sizeof(int));
            checkCUDAErrorFn("cudaMemset d_odata + newSize - 1 failed.");

            for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
                int activeThreads = n >> (d + 1);
                int numBlocks = (activeThreads + threadsPerBlock - 1) / threadsPerBlock;
                kernDownSweep <<<numBlocks, threadsPerBlock>>> (activeThreads, d, data);
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int* odata, const int* idata) {
            size_t newSize = 1 << ilog2ceil(n);

            cudaMalloc((void**)&d_odata, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_odata failed.");
            cudaMemset(d_odata, 0, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMemset d_odata failed.");
            cudaMemcpy(d_odata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy d_odata <- idata failed.");

            timer().startGpuTimer();

            scanWithoutTimer(newSize, d_odata);

            timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, sizeof(int) * n, cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy odata <- d_odata failed.");
            cudaFree(d_odata);
            checkCUDAErrorFn("cudaFree d_odata failed.");
        }
        
        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            size_t newSize = 1 << ilog2ceil(n);

            cudaMalloc((void**)&d_idata, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_idata failed.");
            cudaMalloc((void**)&d_odata, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_odata failed.");
            cudaMalloc((void**)&d_bools, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_bools failed.");
            cudaMalloc((void**)&d_indices, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMalloc d_indices failed.");

            cudaMemset(d_idata, 0, newSize * sizeof(int));
            checkCUDAErrorFn("cudaMemset d_idata failed.");
            cudaMemcpy(d_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            checkCUDAErrorFn("cudaMemcpy d_idata <- idata failed.");

            const int threadsPerBlock = 256;
            int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;

            timer().startGpuTimer();
        
            StreamCompaction::Common::kernMapToBoolean <<<numBlocks, threadsPerBlock>>> (newSize, d_bools, d_idata);
            cudaMemcpy(d_indices, d_bools, newSize * sizeof(int), cudaMemcpyDeviceToDevice);
            checkCUDAErrorFn("cudaMemcpy d_indices <- d_bools failed.");
            scanWithoutTimer(newSize, d_indices);
            StreamCompaction::Common::kernScatter <<<numBlocks, threadsPerBlock>>> (newSize, d_odata, d_idata, d_bools, d_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, d_odata, newSize * sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy odata <- d_odata failed.");

            int count = 0;
            cudaMemcpy(&count, d_indices + newSize - 1, sizeof(int), cudaMemcpyDeviceToHost);
            checkCUDAErrorFn("cudaMemcpy count <- d_indices + newSize - 1 failed.");

            cudaFree(d_idata);
            checkCUDAErrorFn("cudaFree d_idata failed.");
            cudaFree(d_odata);
            checkCUDAErrorFn("cudaFree d_odata failed.");
            cudaFree(d_bools);
            checkCUDAErrorFn("cudaFree d_bools failed.");
            cudaFree(d_indices);
            checkCUDAErrorFn("cudaFree d_indices failed.");

            return count;
        }
    }
}
