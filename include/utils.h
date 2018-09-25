/*
@author:chenzhengqiang
@date:2018-09-25
@email:642346572@qq.com
*/

#ifndef _BACKER_UTILS_H_
#define _BACKER_UTILS_H_
#include "config.h"
#include "buffer.h"

void parse_config( const char * config_file, CONFIG::SERVER_CONFIG & server_config );
void init_global_buffer(BUFFER::GLOBAL_BUFFER & global_buffer, CONFIG::SERVER_CONFIG & server_config);
void free_global_buffer(BUFFER::GLOBAL_BUFFER & global_buffer);

#endif
