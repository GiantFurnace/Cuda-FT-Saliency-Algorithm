/*
@author:chenzhengqiang
@date:2018-06-19
*/


#include "config.h"
#include "opencv_common.h"
#include "gpu_buffer.h"
#include "saliency.h"
#include <iostream>
#include <sys/time.h>


using std::cout;
using std::endl;
using std::cerr;


GPU_BUFFER gpu_buffer;

int main(int argc, char ** argv) 
{

    if( cuda::getCudaEnabledDeviceCount()==0 )
    {
        cerr<<"Fatal error:gpu device unsupported here"<<endl;
        return -1;
    }
    
    if( argc !=2 )
    {
        cerr<<"You must specify one input image"<<endl;
        return -1;
    }

    Mat sample_image = imread(argv[1]);
    if ( sample_image.rows != sample_image.cols || sample_image.rows != GLOBAL_CONFIG::IMAGE_ROWS 
         || sample_image.cols != GLOBAL_CONFIG::IMAGE_COLS )
    {
        cerr<<"Image size unsupported,resize the image before detecting."<<endl;
        return -1;
    }
    
    gpu_buffer.Limg = cuda::GpuMat(GLOBAL_CONFIG::IMAGE_ROWS, GLOBAL_CONFIG::IMAGE_COLS, CV_64FC1);   
    gpu_buffer.Aimg = cuda::GpuMat(GLOBAL_CONFIG::IMAGE_ROWS, GLOBAL_CONFIG::IMAGE_COLS, CV_64FC1);   
    gpu_buffer.Bimg = cuda::GpuMat(GLOBAL_CONFIG::IMAGE_ROWS, GLOBAL_CONFIG::IMAGE_COLS, CV_64FC1);
      
    struct timeval start, end;
    gpu_buffer.sample_image.upload(sample_image);
    cuda::split(gpu_buffer.sample_image, gpu_buffer.BGR);
    gettimeofday(&start, NULL);   
    obtain_saliency_with_ft(gpu_buffer);
    gettimeofday(&end, NULL);
    Mat result;
    gpu_buffer.saliency_map.download(result);
    imwrite("saliency.jpg", result);
    cout<<"gpu took "<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)*0.001<<" ms"<<endl;
    return 0;
}
