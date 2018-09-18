#include "config.h"
#include "saliency.h"
#include "opencv_common.h"
#include "cuda_common.h"


__global__ void bgr2lab(cuda::PtrStepSz<uchar> Bimg_, cuda::PtrStepSz<uchar> Gimg_, cuda::PtrStepSz<uchar> Rimg_,
                        cuda::PtrStepSz<double> Limg, cuda::PtrStepSz<double> Aimg, cuda::PtrStepSz<double> Bimg)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;  
    int y = threadIdx.y + blockIdx.y * blockDim.y; 

    if ( x < Rimg_.rows && y < Rimg_.cols )
    {
        double sB =  static_cast<double>(Bimg_(x,y));
	double sG =  static_cast<double>(Gimg_(x,y));
	double sR =  static_cast<double>(Rimg_(x,y));
	
	double R = sR/255.0;
	double G = sG/255.0;
	double B = sB/255.0;

	double r, g, b;

	if(R <= 0.04045)   r = (double)R/12.92;
	else		   r = pow((R+0.055)/1.055,2.4);
	if(G <= 0.04045)   g = (double)G/12.92;
	else		   g = pow((G+0.055)/1.055,2.4);
	if(B <= 0.04045)   b = (double)B/12.92;
	else		   b = pow((B+0.055)/1.055,2.4);
	
        
	double X = r*0.4124564 + g*0.3575761 + b*0.1804375;
	double Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
	double Z = r*0.0193339 + g*0.1191920 + b*0.9503041;
			
			
	//------------------------
	// XYZ to LAB conversion
	//------------------------
	
	double epsilon = 0.008856; // actual CIE standard
	double kappa   = 903.3;	  // actual CIE standard

	double Xr = 0.950456;	 // reference white
	double Yr = 1.0;		 // reference white
	double Zr = 1.088754;	// reference white

	double xr = X/Xr;
	double yr = Y/Yr;
	double zr = Z/Zr;

	double fx, fy, fz;
	if(xr > epsilon) fx = pow(xr, 1.0/3.0);
	else				fx = (kappa*xr + 16.0)/116.0;
	if(yr > epsilon) fy = pow(yr, 1.0/3.0);
	else				fy = (kappa*yr + 16.0)/116.0;
	if(zr > epsilon) fz = pow(zr, 1.0/3.0);
	else		 fz = (kappa*zr + 16.0)/116.0;

	Limg(x,y) =  yr > epsilon ? (116.0*fy-16.0):(yr*kappa);
	Aimg(x,y)= 500.0*(fx-fy);
	Bimg(x,y)= 200.0*(fy-fz);
    }
    	
}


void obtain_saliency_with_ft( struct GPU_BUFFER & gpu_buffer_ )
{
 		
  	  
    dim3 threads(GLOBAL_CONFIG::CUDA_THREADS_SIZE, GLOBAL_CONFIG::CUDA_THREADS_SIZE);
    dim3 blocks(GLOBAL_CONFIG::IMAGE_ROWS/GLOBAL_CONFIG::CUDA_THREADS_SIZE, GLOBAL_CONFIG::IMAGE_COLS/GLOBAL_CONFIG::CUDA_THREADS_SIZE);
         	
    bgr2lab<<<blocks,threads>>>(gpu_buffer_.BGR[0], gpu_buffer_.BGR[1], gpu_buffer_.BGR[2], 
                                gpu_buffer_.Limg, gpu_buffer_.Aimg, gpu_buffer_.Bimg);
    
    double mean_lab[3];
    
    mean_lab[0] = (cuda::sum(gpu_buffer_.Limg)).val[0] / (GLOBAL_CONFIG::IMAGE_PIXELS);
    mean_lab[1] = (cuda::sum(gpu_buffer_.Aimg)).val[0] / (GLOBAL_CONFIG::IMAGE_PIXELS);
    mean_lab[2] = (cuda::sum(gpu_buffer_.Bimg)).val[0] / (GLOBAL_CONFIG::IMAGE_PIXELS);

    cuda::Stream stream;	
    cuda::absdiff(gpu_buffer_.Limg, mean_lab[0],  gpu_buffer_.Limg, stream);
    cuda::pow(gpu_buffer_.Limg, 2,  gpu_buffer_.Limg, stream);

    cuda::absdiff(gpu_buffer_.Aimg, mean_lab[1],  gpu_buffer_.Aimg, stream);
    cuda::pow(gpu_buffer_.Aimg, 2, gpu_buffer_.Aimg, stream);

    cuda::absdiff(gpu_buffer_.Bimg, mean_lab[2],  gpu_buffer_.Bimg, stream);
    cuda::pow(gpu_buffer_.Bimg, 2, gpu_buffer_.Bimg, stream);

    cuda::add(gpu_buffer_.Limg, gpu_buffer_.Aimg, gpu_buffer_.combine, gpu_buffer_.tmp, -1, stream);
    cuda::add(gpu_buffer_.Bimg, gpu_buffer_.combine, gpu_buffer_.combine, gpu_buffer_.tmp, -1, stream);
    cuda::normalize(gpu_buffer_.combine, gpu_buffer_.combine, 1, 0, NORM_MINMAX, -1, gpu_buffer_.tmp, stream);
    gpu_buffer_.combine.convertTo(gpu_buffer_.saliency_map, CV_8UC1,  255, 0, stream);
    stream.waitForCompletion();
}

