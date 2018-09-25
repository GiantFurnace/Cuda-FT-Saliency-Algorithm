/*
@author:chenzhengqiang
@date:2018-06-19
*/


#include "config.h"
#include "buffer.h"
#include "utils.h"
#include "opencv_common.h"
#include "fabric_detect.h"
#include <iostream>
#include <sys/time.h>


using std::cout;
using std::endl;
using std::cerr;


BUFFER::GLOBAL_BUFFER global_buffer;
const char * DEFAULT_CONFIG_FILE="./config/server.conf";

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
    
    CONFIG::SERVER_CONFIG server_config;
    parse_config(DEFAULT_CONFIG_FILE, server_config);
    init_global_buffer(global_buffer, server_config);
    
    Mat sample_image = imread(argv[1]);
    if ( sample_image.rows != sample_image.cols 
         || sample_image.rows != global_buffer.IMAGE_ROWS 
         || sample_image.cols != global_buffer.IMAGE_COLS )
    {
        cerr<<"Image size unsupported,resize the image before detecting."<<endl;
        return -1;
    }
    
      
    struct timeval start, end;
    gettimeofday(&start, NULL);
    int ret = fabric_detect(sample_image, global_buffer);   
    gettimeofday(&end, NULL);
    cout<<"gpu took "<<(end.tv_sec-start.tv_sec)*1000+(end.tv_usec-start.tv_usec)*0.001<<" ms"<<endl;
    free_global_buffer(global_buffer);
    if ( ret == 0 )
    {
         cout<<"there is no fabric in surface"<<endl;
    }
    else
    {
         cout<<"fabric detected."<<endl;
    }

    return 0;
}
