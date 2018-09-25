/*
@author:chenzhengqiang
@date:2018-09-25
@email:642346572@qq.com
*/


#include "config.h"
#include "utils.h"
#include "opencv_common.h"
#include "cuda_common.h"
#include <cstring>
#include <fstream>
#include <cstdlib>
#include <iostream>


void parse_config( const char * config_file, CONFIG::SERVER_CONFIG & server_config )
{
      std::ifstream ifile( config_file );
      if( ! ifile )
      {
        return;
      }

      std::string line;
      std::string::size_type cur_pos;
      std::string heading,key,value;
      bool is_reading_meta = false;
      bool is_reading_server = false;
      bool is_reading_usage = false;
      server_config.usage="\r";
	
      while( getline( ifile, line ) )
      {
          if( line.empty() )
	  continue;
          if( line[0] == '#' )
	  continue;
          if( line[0] == '[')
          {
               if( ( cur_pos = line.find_last_of(']') ) == std::string::npos )
               {
                    return;
               }
			
	       heading = line.substr( 1, cur_pos-1 );
	       if( heading == "META" )
	       {
	           if( is_reading_server )
	           {
		       is_reading_server = false;
	           }
	           else if( is_reading_usage )
	           {
		       is_reading_usage = false;
	           }
				
	           is_reading_meta = true;
	   }
  		 
	   else if( heading == "SERVER" )
	   {
  		 
	       if( is_reading_meta )
	       {
		    is_reading_meta =false;
	       }
	       else if( is_reading_usage )
	       {
		    is_reading_usage = false;
	       }
	       is_reading_server = true;
  		     
	   }
	   else if( heading == "USAGE" )
	   {
	       if( is_reading_meta )
	       {
		   is_reading_meta = false;
	       }
	       else if( is_reading_server )
	       {
		   is_reading_server = false;
	       }
	       is_reading_usage = true;
	   }
	}
	else if( line[0] == ' ' )
	{
	     return;
	}
	else
	{
	     if( is_reading_meta || is_reading_server )
	     {
		   cur_pos = line.find_first_of('=');
		   if( cur_pos == std::string::npos )
			continue;
				
		   key = line.substr( 0, cur_pos );
		   value = line.substr( cur_pos+1, line.length()-cur_pos-1 );

                       if( is_reading_meta )
                       {
                           server_config.meta.insert( std::make_pair( key, value ) );
                       }
                       else
                       {
                           server_config.server.insert( std::make_pair( key, value ) );
                       }
                  }
                  else if( is_reading_usage )
                  {
                       server_config.usage+=line+"\n\r";
                  }
	}
    }
}



void init_global_buffer(BUFFER::GLOBAL_BUFFER & global_buffer, CONFIG::SERVER_CONFIG & server_config)
{
    // IMAGE_ROWS, IMAGE_COLS, CUDA_THREADS_SIZE, THRESHOLD_MIN, THRESHOLD_MAX
    if ( 
         server_config.server.count("IMAGE_ROWS") <=0 
         || server_config.server.count("IMAGE_COLS") <=0 
         || server_config.server.count("CUDA_THREADS_SIZE") <=0 
         || server_config.server.count("THRESHOLD_MIN") <=0
         || server_config.server.count("THRESHOLD_MAX") <=0
       )
    {
        std::cerr<<"Invalid Config File:IMAGE_ROWS,IMAGE_COLS,CUDA_THREADS_SIZE,THRESHOLD_MIN,THRESHOLD_MAX are all needed."<<std::endl;
        abort();
    }
                 
    global_buffer.IMAGE_ROWS = atoi((server_config.server["IMAGE_ROWS"]).c_str());
    global_buffer.IMAGE_COLS = atoi((server_config.server["IMAGE_COLS"]).c_str());

    if ( global_buffer.IMAGE_ROWS <=0 || global_buffer.IMAGE_ROWS > CONFIG::IMAGE_RC_MAX )
    {
        std::cerr<<"Invalid Config For IMAGE_ROWS."<<std::endl;
        abort();
    }
  
    if ( global_buffer.IMAGE_COLS <=0 || global_buffer.IMAGE_COLS > CONFIG::IMAGE_RC_MAX )
    {
        std::cerr<<"Invalid config for IMAGE_ROWS."<<std::endl;
        abort();
    }  

    global_buffer.IMAGE_PIXELS = global_buffer.IMAGE_ROWS * global_buffer.IMAGE_COLS;
    global_buffer.CUDA_THREADS_SIZE = atoi((server_config.server["CUDA_THREADS_SIZE"]).c_str());
    if ( global_buffer.CUDA_THREADS_SIZE <=0 || global_buffer.CUDA_THREADS_SIZE > CONFIG::CUDA_THREADS_MAX )
    {
        std::cerr<<"Invalid Config For CUDA THREADS SIZE."<<std::endl;
        abort();  
    }
    
    global_buffer.THRESHOLD_MIN = atof((server_config.server["THRESHOLD_MIN"]).c_str());
    global_buffer.THRESHOLD_MAX = atof((server_config.server["THRESHOLD_MAX"]).c_str());

    if ( global_buffer.THRESHOLD_MIN > global_buffer.THRESHOLD_MAX || global_buffer.THRESHOLD_MIN < 0 || global_buffer.THRESHOLD_MAX < 0 )
    {
       std::cerr<<"Invalid Config For THRESHOLD MIN MAX. "<<std::endl;
       abort();
    }

    global_buffer.Limg = cuda::GpuMat(global_buffer.IMAGE_ROWS, global_buffer.IMAGE_COLS, CV_64FC1);
    global_buffer.Aimg = cuda::GpuMat(global_buffer.IMAGE_ROWS, global_buffer.IMAGE_COLS, CV_64FC1);
    global_buffer.Bimg = cuda::GpuMat(global_buffer.IMAGE_ROWS, global_buffer.IMAGE_COLS, CV_64FC1);
    
    cudaMalloc((void**)&(global_buffer.HISTO), 256 * sizeof(int));
    global_buffer.BLOCKS = dim3((global_buffer.IMAGE_ROWS)/global_buffer.CUDA_THREADS_SIZE, 
                                (global_buffer.IMAGE_COLS)/global_buffer.CUDA_THREADS_SIZE);  

    global_buffer.THREADS = dim3(global_buffer.CUDA_THREADS_SIZE, global_buffer.CUDA_THREADS_SIZE);  

}


   
void free_global_buffer(BUFFER::GLOBAL_BUFFER & global_buffer)
{
    cudaFree(global_buffer.HISTO);
}
