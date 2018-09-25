/*
@author:chenzhengqiang
@date:2019-09-18
@email:642346572@qq.com
*/

#ifndef _BACKER_CONFIG_H_
#define _BACKER_CONFIG_H_

#include <map>
#include <string>

namespace CONFIG
{ 
    
    static const int IMAGE_RC_MAX = 960;
    static const int CUDA_THREADS_MAX = 16;
    struct SERVER_CONFIG 
    { 
        std::map<std::string, std::string> meta; 
        std::map<std::string, std::string> server; 
        std::string usage; 
    };
}

#endif
