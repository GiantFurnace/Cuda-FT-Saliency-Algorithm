/*
@author:chenzhengqiang
@date:2018-09-25
@email:642346572@qq.com
*/

#include "fabric_detect.h"
#include "saliency.h"
#include "cvutils.h"
#include "opencv_common.h"
#include <iostream>

int fabric_detect( cv::Mat sample_image, BUFFER::GLOBAL_BUFFER & global_buffer )
{
    cuda::split(sample_image, global_buffer.BGR);
    obtain_saliency_with_ft(global_buffer);
    int max_entropy = get_max_entropy(global_buffer);
    cuda::threshold(global_buffer.saliency_map, global_buffer.threshold_image, max_entropy, 255, CV_THRESH_BINARY);
    global_buffer.threshold_image.download(global_buffer.local);

    int ret = remove_area_by_adaptive_threshold(global_buffer.local);

    if ( ret < 0 )
    {
        return 0;
    }

    double rate = (double)countNonZero(global_buffer.local) / global_buffer.IMAGE_PIXELS;
    if ( rate <= global_buffer.THRESHOLD_MIN || rate >= global_buffer.THRESHOLD_MAX )
    {
        return 0;
    }

    return 1;
}


