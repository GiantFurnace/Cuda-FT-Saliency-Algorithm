/*
@author:chenzhengqiang
@date:2018-09-18
@desc::global config for gpu buffer
*/

#ifndef _BACKER_GPU_BUFFER_H_
#define _BACKER_GPU_BUFFER_H_

#include "opencv_common.h"
#include "cuda_common.h"

namespace BUFFER
{
    struct GLOBAL_BUFFER
    {
        int IMAGE_ROWS, IMAGE_COLS, IMAGE_PIXELS, CUDA_THREADS_SIZE;
        double THRESHOLD_MIN,THRESHOLD_MAX;
        cuda::GpuMat Limg, Aimg, Bimg;
        cuda::GpuMat BGR[3];
        cuda::GpuMat tmp, norm, saliency_map, threshold_image;
        cv::Mat local;
        int *HISTO;
        dim3 BLOCKS, THREADS;
    };
}
#endif
