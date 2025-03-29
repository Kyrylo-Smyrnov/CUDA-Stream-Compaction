#include <cstdio>
#include "cpu.h"
#include "iostream"
#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is su pposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            odata[0] = 0;
            for (int i = 1; i < n; ++i) {
                odata[i] = odata[i - 1] + idata[i - 1];
            }
            timer().endCpuTimer();

            /*-----------------------*/
            /* Naive scan simulation */
            /*-----------------------*/

            // int* tmp = new int[n];
            // std::copy(idata, idata + n, odata);
            // for (int d = 1; d <= ilog2ceil(n); ++d) {
            //     int step = 1 << (d - 1);
            //     std::copy(odata, odata + n, tmp);
            //     for (int k = 0; k < n; ++k) {
            //         if (k >= step) {
            //             odata[k] = tmp[k] + tmp[k - step];
            //         }
            //     }
            // }
            // for (int i = n - 1; i > 0; --i) {
            //     odata[i] = odata[i - 1];
            // }
            // odata[0] = 0;
            // delete[] tmp;

            /*--------------------------------*/
            /* Work-Efficient scan simulation */
            /*--------------------------------*/

            // int* tmp = new int[n];
            // std::copy(idata, idata + n, odata);
            // for (int d = 0; d <= ilog2ceil(n) - 1; ++d) {
            //     int step = 1 << (d - 1);
            //     std::copy(odata, odata + n, tmp);
            //     for (int k = 0; k < n; k += step) {
            //         int sumStep = 1 << d;
            //         odata[k + step - 1] += tmp[k + sumStep - 1];
            //     }
            // }
            // odata[n - 1] = 0;
            // for (int d = ilog2ceil(n) - 1; d >= 0; --d) {
            //     int step = 1 << (d + 1);
            //     for (int k = 0; k < n; k += step) {
            //         int left = odata[k + step / 2 - 1];
            //         int right = odata[k + step - 1];
            //         odata[k + step / 2 - 1] = right;
            //         odata[k + step - 1] += left;
            //     }
            // }
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int num = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i] != 0) {
                    odata[num] = idata[i];
                    num++;
                }
            }
            timer().endCpuTimer();
            return num;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int* scanned = new int[n];
            scanned[0] = 0;
            /* Due to the timer exclusivity and difference in scan algorithm (line 105),
            the scan is not implemented through a function call. */
            for (int i = 1; i < n; ++i) {
                scanned[i] = scanned[i - 1] + (bool)idata[i - 1];
            }
            int count = 0;
            for (int i = 0; i < n; ++i) {
                if (idata[i]) {
                    odata[scanned[i]] = idata[i];
                    count++;
                }
            }
            delete[] scanned;
            timer().endCpuTimer();
            return count;
        }
    }
}
