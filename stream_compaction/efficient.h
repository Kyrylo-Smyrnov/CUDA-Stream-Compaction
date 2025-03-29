#pragma once

#include "common.h"
#include <cuda_runtime.h>

namespace StreamCompaction {
    namespace Efficient {
        StreamCompaction::Common::PerformanceTimer& timer();

        void scanWithoutTimer(int n, int* data);

        void scan(int n, int* odata, const int* idata);

        int compact(int n, int *odata, const int *idata);
    }
}
