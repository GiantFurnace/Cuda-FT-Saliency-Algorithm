/*
@author:chenzhengqiang
@date:2018-09-18
@desc::global config for gpu buffer
*/

#ifndef _BACKER_GPU_BUFFER_H_
#define _BACKER_GPU_BUFFER_H_
#include "opencv_common.h"

struct GPU_BUFFER
{
    cuda::GpuMat Limg, Aimg, Bimg;
    cuda::GpuMat sample_image;
    cuda::GpuMat BGR[3];
    cuda::GpuMat tmp, combine, saliency_map;
};

#endif
