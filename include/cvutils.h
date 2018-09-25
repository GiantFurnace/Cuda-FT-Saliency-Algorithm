/*
@author:chenzhengqiang
@date:2018-09-25
@email:642346572@qq.com
*/

#ifndef _BACKER_CVUTILS_H_
#define _BACKER_CVUTILS_H_

#include "buffer.h"
int get_max_entropy(BUFFER::GLOBAL_BUFFER & global_buffer);
int remove_area_by_adaptive_threshold(Mat & src_image);
#endif
