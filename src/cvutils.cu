/*
@author:chenzhengqiang
@date:2018-09-25
*/


#include "cvutils.h"
#include "opencv_common.h"
#include "cuda_common.h"
#include <iostream>
#include <vector>
using std::vector;

__global__ void calc_image_histo(const cuda::PtrStepSz<uchar> src_image, int* histo)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ( x < src_image.rows && y < src_image.cols )
    {
        int index = static_cast<int>(src_image(x, y));
        atomicAdd(&(histo[index]), 1);
    }
}


int get_max_entropy( BUFFER::GLOBAL_BUFFER & global_buffer )
{
    cudaMemset(global_buffer.HISTO, 0, 256 * sizeof(int));   
    calc_image_histo<<<global_buffer.BLOCKS, global_buffer.THREADS>>>(global_buffer.saliency_map, global_buffer.HISTO);
    int histo[256]={0};
    cudaMemcpy( histo, global_buffer.HISTO, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost );

    int pixel_index = 0;                                                  
    double property = 0.0;                                         
    double max_entropy = -1.0;                                      
    double front_entropy = 0.0;                                  
    double back_entropy = 0.0;
 
    for (int i = 0; i< 256;  i++) 
    {
        double back_total = 0;
        for (int j = 0; j < i; j++)
        {
            back_total += histo[j];
        }

        for (int j = 0; j < i; j++)
        {
            if (histo[j] != 0)
            {
                property = histo[j] / back_total;
                back_entropy += -property * logf((float)property);
            }
        }
        
        for (int k = i; k < 256; k++)
        {
            if (histo[k] != 0)
            {
                property = histo[k] / (global_buffer.IMAGE_PIXELS - back_total);
                front_entropy += -property * logf((float)property);
            }
        }

        if ((front_entropy + back_entropy) > max_entropy)
        {
            max_entropy = front_entropy + back_entropy;
            pixel_index = i;
        }
       
        front_entropy = 0.0;
        back_entropy = 0.0;
    }
    
    return pixel_index;

}



int remove_area_by_adaptive_threshold(Mat & src_image)
{
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;
    vector<double> areas;  
    findContours(src_image, contours, hierarchy, CV_RETR_LIST, CHAIN_APPROX_NONE, Point(0, 0));
    
    int ret = 0;
    double threshold_area = 0;
    int area_size = (int) contours.size();
    if ( area_size <= 0 )
    return -1;
	
    for (int i = 0; i < area_size; ++i )
    {
         //areas.push_back(contourArea(contours[i], false));
         threshold_area += contourArea(contours[i]);
    }
    
    threshold_area = threshold_area / area_size;
     
    //calculate the max threashold area 
    /*sort(areas.begin(), areas.end());
    double  max_sub = -1; 
    int max_sub_index = 0;

    for (int i = 0; i < areas.size(); i++)
    {
        
	int j = i+1;
        if ( j == areas.size() )
            break; 

	double area_sub = areas[j]-areas[i];
        if ( area_sub > max_sub )
        {
            max_sub_index = j;
            max_sub = area_sub;
        }
              
    }

    threshold_area = areas[max_sub_index];*/
    vector< vector< Point> > contours2;

    vector<vector<Point> >::iterator iter = contours.begin(); 
    while ( iter != contours.end() ) 
    {    
        
	if( contourArea(*iter, false) < threshold_area)
	{  
	    //iter = contours.erase(iter);
            contours2.push_back(*iter);
	}
        ++iter;
	/*else
	{  
            ++ret;
	    ++iter;	
	}*/
    }  
       
    drawContours(src_image, contours2, -1, Scalar(0), CV_FILLED);
    return ret;
}
