#include <vector>
#include <windows.h>
#include<sstream>
#include <math.h>

#include "opencv2/opencv.hpp"
#include "splines.h"
// #include "filter.h"

using namespace std;
using namespace cv;

const double  PI = 3.1415926535897932384626433832795;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float sigma = 0.5;//0.8;
static float gauss_coeff = 1.0/(2*M_PI*sigma*sigma);


/******************����Ӧ�����ص�*******************/
// ����������-1��1���������
#define asinDomain(x) (x > 1 ? 1 : ( x < -1 ? -1 : x ) )




// �ӱ궨ͼ������ͼ��ķ�����ӳ��,�ã�x,y����ʾ��γ����ϵ�϶�Ӧ������(longitude, latitude��,�ã�u,v����ʾ����ͼ���ϵĵ�(x,y)
void nonlinear_mapping_orthogonal(double x,double y,double &u,double &v,double foval, double factor_c = 1 /*У��ͼ�������*/, double factor_f = 1/*����ͼ�������*/)
{

	x = x * factor_c;
	y = y * factor_c;

	double distance = sqrt(pow(x , 2) + pow(y , 2)) ;
	if( abs(distance) <= 1e-15 ){
	   u = 0;
	   v = 0;
	   return;
	}
	
	double Theta_sphere = cvFastArctan(distance, foval) * PI / 180;
	double p = foval * sin( Theta_sphere ); 
	
	u = ( p * x ) / distance;
	v = ( p * y ) / distance;

	u = u * factor_f;
    v = v * factor_f;

}

// ������ͼ�񵽱궨ͼ��ķ�����ӳ��
void inverse_nonlinear_mapping_orthogonal(double u,double v,double &x,double &y,double foval, double factor_f = 1 /*����ͼ�������*/, double factor_c = 1 /*У��ͼ�������*/)
{

	u = u / factor_f;
    v = v / factor_f;

	double p= sqrt(pow(u, 2) + pow(v, 2));
	
	if( abs(p) <= 1e-15 ){
		x = 0;
		y = 0;
	   return;
	}

	double Theta_sphere = asin(asinDomain(p / foval));
	double distance = foval * tan(Theta_sphere);

	x = ( distance * u ) / p; 
	y = ( distance * v ) / p;

	x = x / factor_c;
	y = y / factor_c;

}

// ����ԭͼ����ÿ������Ÿ�������ʽ�ľ���ֵ ע�⣺���ܻ�ȡ0ֵ
double abs_jacobian_Determinant_of_mapping_orthogonal(double x,double y,double foval)
{
	// abs( -( f^4 / (f^2 + x^2 + y^2)^2  ) )
	return abs( pow(foval, 4) / pow(pow(foval, 2) + pow(x, 2) + pow(y, 2), 2));  
}

//ԭͼ����ÿ��������Ȩ��
void areaWeight_orthogonal(vector<float> &S_xy,int width, int height,
						   int center_x, int center_y, float foval, int radius, double factor_f,double factor_c,
						   void (*inverse_mapping)(double, double, double &, double &, double, double, double), 
				double (*abs_jacobian_det)(double, double, double)){
		int x,y;
		int x_temp, y_temp;
		double x_simu, y_simu;

		for (int i = 0; i < width*height; i++){	
			//���������ת��Ϊ���������ϵ��ԭ������Ϊԭ�������
			 x = i % width;
		     y = i / width;

			x_temp = ( x - center_x );
		    y_temp = ( y - center_y );

			inverse_mapping(x_temp, y_temp, x_simu, y_simu, foval, factor_f, factor_c);
			S_xy[i]=abs_jacobian_det(x_simu, y_simu, foval);
	    }

}


// ������ԭʼͼ���Ͻ��о�����������ĳ��Ϳ�
void computeConvArea(int &Horizontalksize, int &Verticalksize,  int x, int y, float sigma,
					 void (*mapping)(double, double, double &, double &, double, double, double), float foval, double factor_c, double factor_f){

			/* ����ģ��ͼ���ϵľ������ */
			double kx[4],ky[4];			
			kx[0] = (float)(x-4 * sigma);
			ky[0] = (float)(y+4 * sigma);
			kx[1] = (float)(x+4 * sigma);
			ky[1] = (float)(y+4 * sigma);
			kx[2] = (float)(x+4 * sigma);
			ky[2] = (float)(y-4 * sigma);
			kx[3] = (float)(x-4 * sigma);
			ky[3] = (float)(y-4 * sigma);
			/**/

			/* ����ԭʼͼ��ľ������ */
			 Horizontalksize = 0,
			   Verticalksize = 0;
			
			for(int i = 0;i<4;i++)
			{	
				mapping(kx[i],ky[i],kx[i],ky[i],foval, factor_c, factor_f);
			}

			//��x��y���������Сֵ
			float kx_min,kx_max,ky_min,ky_max;
			kx_min = kx[0];
			kx_max = kx[0];
			ky_min = ky[0];
			ky_max = ky[0];
			for(int i = 1;i<4;i++)
			{
				kx_min = min((float)kx[i],kx_min);
				kx_max = max((float)kx[i],kx_max);
				ky_min = min((float)ky[i],ky_min);
				ky_max = max((float)ky[i],ky_max);
			}

			int kx_imin,kx_imax,ky_imin,ky_imax;
			kx_imin = (int)floor(kx_min); 
			kx_imax = (int)floor(kx_max);
			ky_imin = (int)floor(ky_min);
			ky_imax = (int)floor(ky_max);

			Horizontalksize=(int)(kx_imax-kx_imin+1);
			Verticalksize=(int)(ky_imax-ky_imin+1);

			// Make kernel size odd. 
			if (Horizontalksize % 2 == 0)       
				Horizontalksize++;			

			if (Verticalksize % 2 == 0)       
				Verticalksize++;
			/**/
}

// ������Ӧ����ͼ����˲���
void fisheye_computeGaussKernel_withconv_withPreProc(Mat &kernel, float sigma, float x_org, float y_org, float x_center, float y_center, int width, int height,
													 int x_simuImg,int y_simuImg,
													 double foval, int radius, double factor_f,double factor_c,
													 void (*inverse_mapping)(double, double, double &, double &, double, double, double), vector<float>& S_xy )
{
	/* (x_simuImg,y_simuImg)Ϊģ��ͼ������Ҫ����ֵ�ĵ������,����ϵԭ����ͼ������ */	
	float kx, ky;
	float sum = 0.0;
	/* Fill in kernel values. */

	float Sigma_Coeff =  2 * sigma * sigma ;

	double simu_kx_temp;
	double simu_ky_temp;

	int Verticalksize = kernel.rows;
	int Horizontalksize = kernel.cols;

	for (int j = 0; j < Verticalksize; j++){
	   for (int i = 0; i < Horizontalksize; i++){

				kx = x_org + i - (Horizontalksize-1) / 2;
				ky = y_org + j - (Verticalksize-1) / 2;

				// λ����Ч������ĵ㣬��Ϊ0������ǰ������һ��				
				if( (kx < 0) || (kx >= width) || (ky < 0) || (ky >= height) ){
					kernel.at<float>(j, i) = 0;
					continue;
				}
			    
				kx = kx - x_center;
				ky = ky - y_center;

				inverse_mapping(kx,ky,simu_kx_temp,simu_ky_temp, foval, factor_f, factor_c);

				simu_kx_temp = simu_kx_temp - x_simuImg;
				simu_ky_temp = simu_ky_temp - y_simuImg;

				/*
				// ����ģ��ͼ�������ξ������ĵ�,������,����Ϊ0
				if (abs( simu_kx_temp ) > conv_radius_stimu  || abs(simu_ky_temp) > conv_radius_stimu){
				    kernel.at<float>(j, i) = 0; 
					continue;
				} 
				*/

				kx = kx + x_center;
				ky = ky + y_center;
			
				float val = S_xy[((int)ky)*width+(int)kx];
				
				// ��ĸ��Ϊ0�������˵�ֵ��Ϊ0������ǰ������һ��				
				if(fabs(val)<1e-8){
					kernel.at<float>(j, i) = 0; 
					continue;
				}
				
				// kernel.at<float>(j, i) =  exp(- (simu_kx_temp * simu_kx_temp + simu_ky_temp * simu_ky_temp)/Sigma_Coeff );
				
			    kernel.at<float>(j, i) = gauss_coeff * exp(-(simu_kx_temp * simu_kx_temp + simu_ky_temp * simu_ky_temp)/Sigma_Coeff ) / val;			
			    sum += kernel.at<float>(j, i);
	      }
	 }
	
	
	 for(int j = 0; j < Verticalksize; j++){
	    for (int i = 0; i < Horizontalksize; i++){
			kernel.at<float>(j, i) = kernel.at<float>(j, i) / sum;
		}
	 } 
	 

}

// ִ�о������
void fisheye_perGaussianBlur2D_withconv(uchar (&convResult),vector<float>& image, 
										int subInter,int width, int height, 
										Mat kernel, int Horizontalksize, int Verticalksize){

	int x_subInter = subInter % width;
	int y_subInter = subInter / width;

	convResult = 0;

	if(x_subInter < 0 || x_subInter >= width || y_subInter < 0 || y_subInter >= height )
		return;
		
	int convx,convy;
	int radiusX = (Horizontalksize - 1) / 2;
	int radiusY = (Verticalksize - 1) / 2;

	for (int j = 0; j < Verticalksize; j++){
	    for (int i = 0; i < Horizontalksize; i++){

			    convy = y_subInter + radiusY - j;
				convx = x_subInter + radiusX - i;

				if(convx>=0 && convx<width && convy>=0 && convy<height){  					
					convResult += (kernel.at<float>(j, i) * image[convy*width+convx]);
				}
		}	
	}
}

//**��ֵ�������*///
// �������ڲ�ֵ������ͼ��
void generateInterpolatedView(vector<float> &ref, float *ak,
							  vector<float> view, int imgWidth, int imgHeight, int order){

	vector<float> _coeffs;	
	int size = imgWidth * imgHeight;
	if (order >= 3) {
		_coeffs = vector<float>( size );

		finvspline(view, order, _coeffs, imgWidth, imgHeight);
		ref = _coeffs;

		if (order > 3) init_splinen(ak, order);
	} else {
		ref = view;
	}
}

// �����ֵ������ϵ��
void coefficientsOfHighOrderIntepolation(int &intervalStart,int &intervalEnd, float *cx, float *cy, 
										 float ux, float uy, float paramOfBicubic, float *ak,
										 int order){
			switch (order) 
			{
			case 1: /* first order interpolation (bilinear) */
				intervalEnd = 1;
				cx[0]=ux; cx[1]=1.-ux;
				cy[0]=uy; cy[1]=1.-uy;
				break;

			case -3: /* third order interpolation (bicubic Keys' function) */
				intervalEnd = 2;
				keys(cx, ux, paramOfBicubic);
				keys(cy, uy, paramOfBicubic);
				break;

			case 3: /* spline of order 3 */
				intervalEnd = 2;
				spline3(cx, ux);
				spline3(cy, uy);
				break;

			default: /* spline of order >3 */
				intervalEnd = (1 + order) / 2;
				splinen(cx, ux, ak, order);
				splinen(cy, uy, ak, order);
				break;
			}
           intervalStart = 1 - intervalEnd;

	}


void getInsertedVal(float &res, double xp, double yp, 
					vector<float> originalView, int imgHeight, int imgWidth, vector<float> ref, 
					float paramOfBicubic = 0, float bgColor = 128.0, int order = 1){

	int  xi,yi;
	// ��ֵ�����������յ�
	int  intervalStart, intervalEnd;
	// adr Ҫ��ֵ�ĵ�, dxΪ��ֵ�������x�����ƫ����, dyΪ��ֵ�������y�����ƫ����
	int  adr,dx,dy; 
	float  ux,uy;
	float  cx[12],cy[12],ak[13];


	// ������Χ�ĵ�����Ϊ����
	if (xp<0. || (int)xp >= imgWidth || yp<0. || (int)yp >= imgHeight ){
	    res = bgColor;
	} 

	//��ֵ��ʼ
	xi = (int)floor( (double)xp ); 
	yi = (int)floor( (double)yp );

	/* zero order interpolation (pixel replication) */
	if( order == 0 ){
		res = originalView[ yi * imgWidth + xi ];
		return;
	}

	ux = xp - (float)xi;
	uy = yp - (float)yi;
	
	coefficientsOfHighOrderIntepolation(intervalStart,intervalEnd, cx, cy, ux, uy, paramOfBicubic, ak, order);

	res = 0.; 
	/* this test saves computation time */
	if (xi + intervalStart >= 0 && xi + intervalEnd < imgWidth 
		&& yi + intervalStart >= 0 && yi + intervalEnd < imgHeight) {
		adr = yi*imgWidth + xi;

		for (dy = intervalStart;dy <= intervalEnd;dy++){
			for (dx = intervalStart;dx <= intervalEnd;dx++){
					res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * ref[adr + imgWidth * dy + dx];
			}
		}

	}else{

		for (dy = intervalStart;dy <= intervalEnd;dy++){
			for (dx = intervalStart;dx <= intervalEnd;dx++){ 
					res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * v( ref, xi + dx, yi + dy, bgColor, imgWidth, imgHeight); 
			}
		}

	}

}


/*********************************************/


/*********************��ͨ�ĸ�˹����************************/
// �����˹�˲���
void computeGaussKernel(Mat &kernel, float sigma){
  //(x_simuImg,y_simuImg)Ϊģ��ͼ������Ҫ����ֵ�ĵ������,����ϵԭ����ͼ������ 	
	
	
	
	float kx, ky;
	float sum = 0.0;
// Fill in kernel values. 

	float Sigma_Coeff = 2.0 * sigma * sigma;

	int ksize = (int)(2.0 * 4.0 * sigma + 1.0);
    float shift = 0.5 * (float)(ksize - 1);

	kernel = Mat(ksize,ksize,CV_32FC1);

	float val = 0.0;

	for(int h = 0 ;h <= shift ; h++){
		for(int w = 0 ;w <= shift; w++){
		    
			ky =  h - shift;
			kx =  w - shift;
			


		    kernel.at<float>(h, w)
				= kernel.at<float>(h,ksize - 1 - w)
				= kernel.at<float>(ksize - 1 - h, w)
				= kernel.at<float>(ksize - 1 - h,ksize - 1 - w)
				= gauss_coeff * exp(- (kx * kx + ky * ky)/Sigma_Coeff );

			sum += (kernel.at<float>(h, w) * 4); 
		
		}
	}


	
	for(int h = 0 ;h <= shift ; h++){
		for(int w = 0 ;w <= shift; w++){
		    
			val = kernel.at<float>(h, w) / sum;
			
		    kernel.at<float>(h, w) = val;
		    kernel.at<float>(h,ksize - 1 - w) = val;
		    kernel.at<float>(ksize - 1 - h, w) = val;
			kernel.at<float>(ksize - 1 - h,ksize - 1 - w) = val;
					
		}
	}

	
	

}

void myGaussian(const Mat _src, Mat &_dst,Mat kernel) {  
   
	if (!_src.data) return;  

    Mat tmp(_src.size(), _src.type());  
	
	for(int j = 0 ; j < _src.rows ; j++){
		for(int i = 0 ; i<_src.cols ; i++){

		    if(i < 0 || i >= _src.cols || j < 0 || j >= _src.rows )
				continue;

			tmp.at<uchar>(j, i) = 0;

			int Horizontalksize = kernel.rows;
			int Verticalksize = kernel.cols;
			int radiusX = (Horizontalksize - 1) / 2;
			int radiusY = (Verticalksize - 1) / 2;
			int convx, convy;

			for (int y = 0; y < Verticalksize; ++y) {    
				    for (int x = 0; x < Horizontalksize; ++x) {  
                     
						convy = j + radiusY - y;
						convx = i + radiusX - x;
					
					if(convx < 0 || convx >= _src.cols || convy < 0 || convy >= _src.rows)
					    continue;

					tmp.at<uchar>(j, i) += ( kernel.at<float>(y, x) * _src.at<uchar>(convy, convx) );  		    
				}  
            }
		}
	}

    tmp.copyTo(_dst);  
}  

/*********************************************/


/*****************��������****************************/

// ������ת������
void matToVec(Mat m, vector<float> &vec){

	
	for(int j = 0;j < m.rows;j++){
	    for(int i = 0;i < m.cols;i++){
			vec.push_back(m.at<uchar>(j,i));		
		}
	}
}

void matToVec_float(Mat m, vector<float> &vec){

	for(int j = 0;j < m.rows;j++){
	    for(int i = 0;i < m.cols;i++){
			vec.push_back(m.at<float>(j,i));		
		}
	}

}

/*******************���ڿ��ӻ�**************************/
// ��ǿʵ��Ľ���Ч��
void enhancementEffect(Mat &img){

	float maxValue = 0;
	int imgHeight = img.rows,
		imgWidth = img.cols;

	for(int h = 0;h < imgHeight;h++){
	    for(int w = 0;w < imgWidth;w++){		
			if(img.at<uchar>(h,w) > maxValue){
			   maxValue =  img.at<uchar>(h,w);
			} 
		}
	}
	
	for(int h = 0;h < imgHeight;h++){
	    for(int w = 0;w < imgWidth;w++){	
			   img.at<uchar>(h,w) = img.at<uchar>(h,w) * 255 / maxValue;
		}
	}
}

void enhancementEffect_float(Mat &img){

	float maxValue = 0;
	int imgHeight = img.rows,
		imgWidth = img.cols;

	for(int h = 0;h < imgHeight;h++){
	    for(int w = 0;w < imgWidth;w++){		
			if(img.at<float>(h,w) > maxValue){
			   maxValue =  img.at<float>(h,w);
			} 
		}
	}
	
	for(int h = 0;h < imgHeight;h++){
	    for(int w = 0;w < imgWidth;w++){	
			   img.at<float>(h,w) = img.at<float>(h,w) * 255 / maxValue;
		}
	}
}


// ������������ͼ
void generateFrontView(Mat &frontViewImg,string fileName,
					   int imgHeight = 64,int imgWidth = 64, int blockW = 1, int blockH = 1, int blockNum = 3){

	int StepY = imgHeight / (blockNum + 1);
    int StepX =  imgWidth / (blockNum + 1); 
	int xPos = 0;
	int yPos = 0;

	frontViewImg = Mat::zeros(imgHeight, imgWidth, CV_8UC1); 

	for(int j = 1 ; j <= blockNum; j++){
		yPos = j * StepY;
		for(int i = 1 ; i <= blockNum; i++){
			xPos = i * StepX;			
			for(int offSetY = - blockH;offSetY <= blockH; offSetY++)
				for(int offSetX = - blockW;offSetX <= blockW; offSetX++)
		             frontViewImg.at<uchar>(yPos + offSetY,xPos + offSetX) = 255;
		
		}
	}

	cv::imwrite(fileName, frontViewImg);


}

// ��������ģ��ͼ��
void generateGaussianBlurView(Mat viewImg, Mat &blurViewImg, double sigma, string fileName){

	 int ksize = (int)(2.0 * 4.0 * sigma + 1.0);
	 cv::GaussianBlur(viewImg, blurViewImg, Size(ksize, ksize), sigma);

	enhancementEffect(blurViewImg);

	cv::imwrite(fileName, blurViewImg);

}

void generateGaussianBlurView_float(Mat viewImg, Mat &blurViewImg, double sigma, string fileName){

	 int ksize = (int)(2.0 * 4.0 * sigma);

	 ksize = ( ksize % 2 == 0 ?  ++ksize : ksize );

	 ksize;
	 cv::GaussianBlur(viewImg, blurViewImg, Size(ksize, ksize), sigma);

	enhancementEffect_float(blurViewImg);

	cv::imwrite(fileName, blurViewImg);

}
/*********************************************/



// ������������ͼ��
void generateFisheyeView(Mat &fisheyeViewImg,  int radius, double camerFieldAngle, Mat viewImg, string fileName, double factor = 1 /* ����ͼ��ķ�������*/ ){

	int imgHeight = viewImg.rows;
	int imgWidth = viewImg.cols;
	double foval = 0.0;//����
	 // ��������ͶӰģ�ͼ���������� R_max = f * sin(theta_max), fovalΪ����
	foval = radius / sin(camerFieldAngle / 2);

	Point2i center;
	center.y = (imgHeight - 1) / 2;
	center.x = (imgWidth - 1) / 2;

	int fisheyeImgHeight = imgHeight * factor;
	int fisheyeImgWidth = imgWidth * factor;

	Point2i center_fisheye;
	center_fisheye.y = (fisheyeImgHeight - 1) / 2;
	center_fisheye.x = (fisheyeImgWidth - 1) / 2;
	
	Mat _fisheyeImg = Mat::zeros(fisheyeImgHeight, fisheyeImgWidth, CV_32FC1);

	int fex,fey;
	double xp,yp;

	int size = imgHeight * imgWidth;
	vector<float> frontView;

	matToVec_float(viewImg, frontView);

	


	/* INTERPOLATION */
	  int order = 3;
	float bgColor = 0.0;  // ��������            
	float paramOfBicubic = 0;  // float fperproj_p = 0; float *p = &fperproj_p;

	int  xi,yi;
	// ��ֵ�����������յ�
	int  intervalStart, intervalEnd;
	// adr Ҫ��ֵ�ĵ�, dxΪ��ֵ�������x�����ƫ����, dyΪ��ֵ�������y�����ƫ����
	int  adr,dx,dy; 
	float  res,ux,uy;
	float  cx[12],cy[12],ak[13];

	vector<float> ref, coeffs;

	generateInterpolatedView(ref, ak, frontView, imgWidth, imgHeight, order);


	for(int j = 0;j < fisheyeImgHeight;j++){
		for(int i = 0;i< fisheyeImgWidth;i++){
            
			fey = (j - center_fisheye.y); 
			fex = (i - center_fisheye.x); 

			inverse_nonlinear_mapping_orthogonal(fex, fey, xp, yp, foval, factor);

			yp = yp + center.y;
			xp = xp + center.x;

			// ������Χ�ĵ�����Ϊ����
			if (xp<0. || (int)xp >= imgWidth || yp<0. || (int)yp >= imgHeight )
			{ 
					res = bgColor;
			} 

			//��ֵ��ʼ
			//jzx �޸� ����Ҫ����0.5
			//xp -= 0.5; yp -= 0.5;
			xi = (int)floor( (double)xp ); 
			yi = (int)floor( (double)yp );

			/* zero order interpolation (pixel replication) */
			if( order == 0 ){
				res = frontView[ yi * imgWidth + xi ];
				continue;
			}

			ux = xp - (float)xi;
			uy = yp - (float)yi;
	
	coefficientsOfHighOrderIntepolation(intervalStart,intervalEnd, cx, cy, ux, uy, paramOfBicubic, ak, order);

			res = 0.; 
			/* this test saves computation time */

			


			if (xi > 0 && yi > 0 && 
				xi + intervalStart >= 0 && xi + intervalEnd < imgWidth && 
				yi + intervalStart >= 0 && yi + intervalEnd < imgHeight  ) {

				adr = yi*imgWidth + xi;
				
				for (dy = intervalStart;dy <= intervalEnd;dy++){
					for (dx = intervalStart;dx <= intervalEnd;dx++){
   
						  res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * ref[adr + imgWidth * dy + dx];
					}
				}
			} 
			else 
			{
				for (dy = intervalStart;dy <= intervalEnd;dy++){
					for (dx = intervalStart;dx <= intervalEnd;dx++){ 
							res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * v( ref, xi + dx, yi + dy, bgColor, imgWidth, imgHeight); 
					}
				}
			}

			_fisheyeImg.at<float>(j,i) =  res;
		}
	}

	fisheyeViewImg = _fisheyeImg;

	cv::imwrite(fileName, _fisheyeImg);
}

// ��������У������ͼ��
void generateCorrectedView(Mat &correctedViewImg, int correctedImgHeight,int correctedImgWidth, int radius, double camerFieldAngle, Mat fisheyeImg, string fileName, float  factor_f = 1 /* ����ͼ��ķ�������*/, float  factor_c = 1 /* У��ͼ��ķ�������*/ ){
  
	

	correctedImgHeight = correctedImgHeight  / factor_c;
	correctedImgWidth = correctedImgWidth  / factor_c;

	Point2i center;
	center.y = (correctedImgHeight - 1) / 2;
	center.x = (correctedImgWidth - 1) / 2;
	
	Mat correctedImg(correctedImgHeight, correctedImgWidth, CV_32FC1); 

	int fisheyeImgHeight = fisheyeImg.rows;
	int fisheyeImgWidth = fisheyeImg.cols;


	Point2i center_fisheye;
	center_fisheye.y = (fisheyeImgHeight - 1) / 2;
	center_fisheye.x = (fisheyeImgWidth - 1) / 2;
	
	double foval = 0.0;//����
	 // ��������ͶӰģ�ͼ���������� R_max = f * sin(theta_max), fovalΪ����
	foval = radius / sin(camerFieldAngle / 2);
 	
	vector<float> fisheyeView;

	matToVec_float(fisheyeImg, fisheyeView);

	/* INTERPOLATION */
	  int order = 3;
	float bgColor = 0.0;  // ��������            
	float paramOfBicubic = 0;  // float fperproj_p = 0; float *p = &fperproj_p;

	int  xi,yi;
	// ��ֵ�����������յ�
	int  intervalStart, intervalEnd;
	// adr Ҫ��ֵ�ĵ�, dxΪ��ֵ�������x�����ƫ����, dyΪ��ֵ�������y�����ƫ����
	int  adr,dx,dy; 
	float  res,ux,uy;
	float  cx[12],cy[12],ak[13];

	vector<float> ref, coeffs;


	uchar convResult = 0;
	generateInterpolatedView(ref, ak, fisheyeView, fisheyeImgWidth, fisheyeImgHeight, order);

	

	int cdx,cdy;
	double xp,yp;
	
	for(int j = 0;j <  correctedImgHeight;j++){
		for(int i = 0;i< correctedImgWidth;i++){
            
			cdy = (j - center.y); 
			cdx = (i - center.x); 

			nonlinear_mapping_orthogonal(cdx, cdy, xp, yp, foval, factor_c, factor_f);

			yp = (yp  + center_fisheye.y) ;
			xp = (xp  + center_fisheye.x) ;

			// ������Χ�ĵ�����Ϊ����
			if (xp < 0. || (int)xp >= fisheyeImgWidth || yp < 0. || (int)yp >= fisheyeImgHeight )
			{ 
					res = bgColor;
					continue;
			} 
        
			//��ֵ��ʼ
			//jzx �޸� ����Ҫ����0.5
			//xp -= 0.5; yp -= 0.5;
			xi = (int)floor( (double)xp ); 
			yi = (int)floor( (double)yp );

			/* ��ײ�ֵ */
			/* zero order interpolation (pixel replication) */
			if( order == 0 ){
				correctedImg.at<float>(j,i) = fisheyeView[ yi * fisheyeImgWidth + xi ];
				continue;
			}

			/* �߽ײ�ֵ */
			ux = xp - (float)xi;
			uy = yp - (float)yi;
	
			coefficientsOfHighOrderIntepolation(intervalStart,intervalEnd, cx, cy, ux, uy, paramOfBicubic, ak, order);

	         res = 0.;
			/* this test saves computation time */
			if (xi > 0 && yi > 0 &&
				xi + intervalStart >= 0 && xi + intervalEnd < fisheyeImgWidth && 
				yi + intervalStart >= 0 && yi + intervalEnd < fisheyeImgHeight) {

				adr = yi*fisheyeImgWidth + xi; 
				
				for (dy = intervalStart;dy <= intervalEnd;dy++){
					for (dx = intervalStart;dx <= intervalEnd;dx++){

						 res +=  cy[intervalEnd - dy] * cx[intervalEnd - dx] * ref[ adr+fisheyeImgWidth*dy+dx ];
					}
				}
			 } 
			 else 
			 {
			 	for (dy = intervalStart;dy <= intervalEnd;dy++){
			 		for (dx = intervalStart;dx <= intervalEnd;dx++){ 

			 				res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * v( ref, xi + dx, yi + dy, bgColor, fisheyeImgWidth, fisheyeImgHeight); 
			 		}
			 	}
			 } 

			correctedImg.at<float>(j,i) =  res;
			//correctedImg.at<uchar>(j,i) = fisheyeImg.at<uchar>(yp,xp);
		}
	}

	enhancementEffect_float(correctedImg);
	correctedViewImg = correctedImg;

	cv::imwrite(fileName, correctedImg);
	/*
	Size dsize = Size(correctedImg.cols * 0.5, correctedImg.rows * 0.5);  
     Mat imagedst = Mat(dsize, CV_32S);  
    resize(correctedImg, imagedst, dsize);

	correctedViewImg = imagedst;
	cv::imwrite(fileName, imagedst);
	*/

}



// ������ͼ�����MI���
//
//void generatefisheyeViewWithMIConv(Mat &fisheyeMIBlurImg, int correctedImgHeight,int correctedImgWidth, int radius, double camerFieldAngle, Mat fisheyeImg, double sigma, string fileName, float  factor_f = 1 /* ����ͼ��ķ�������*/, float  factor_c = 1 /* У��ͼ��ķ�������*/){
//  
//	Point2i center;
//	center.y = (correctedImgHeight - 1) / 2;
//	center.x = (correctedImgWidth - 1) / 2;
//	
//	int fisheyeImgHeight = fisheyeImg.rows;
//	int fisheyeImgWidth = fisheyeImg.cols;
//
//	Point2i center_fisheye;
//	center_fisheye.y = (fisheyeImgHeight - 1) / 2;
//	center_fisheye.x = (fisheyeImgWidth - 1) / 2;
//
//	fisheyeMIBlurImg = Mat(fisheyeImgHeight, fisheyeImgWidth, CV_32FC1);
//
//	double foval = 0.0;//����
//	 // ��������ͶӰģ�ͼ���������� R_max = f * sin(theta_max), fovalΪ����
//	foval = radius / sin(camerFieldAngle / 2);
// 	
//	vector<float> fisheyeView;
//
//	matToVec_float(fisheyeImg, fisheyeView);
//
//	/* INTERPOLATION */
//	  int order = 0;
//	float bgColor = 0.0;  // ��������            
//	float paramOfBicubic = 0;  // float fperproj_p = 0; float *p = &fperproj_p;
//
//	int  xi,yi;
//	// ��ֵ�����������յ�
//	int  intervalStart, intervalEnd;
//	// adr Ҫ��ֵ�ĵ�, dxΪ��ֵ�������x�����ƫ����, dyΪ��ֵ�������y�����ƫ����
//	int  adr,dx,dy; 
//	float  res,ux,uy;
//	float  cx[12],cy[12],ak[13];
//
//	vector<float> ref, coeffs;
//
//	uchar convResult = 0;
//	generateInterpolatedView(ref, ak, fisheyeView, fisheyeImgWidth, fisheyeImgHeight, order);
//
//	int cdx,cdy;
//	double xp,yp;
//
//
//#pragma region ����ԭͼ����ÿ�����Ȩ��
//
//	// Ϊ��������ʼ��ֵ����ֹ����ʱ����
//   vector<float> S_xy(fisheyeImgWidth * fisheyeImgHeight, 0);
//   //����ԭͼ����ÿ��������Ȩ��,ÿ��ģ��ͼ���Ӧһ�����Ȩ��
//	areaWeight_orthogonal(S_xy, fisheyeImgWidth, fisheyeImgHeight, center_fisheye.x, center_fisheye.y, foval, radius, factor_f,factor_c,
//		inverse_nonlinear_mapping_orthogonal, abs_jacobian_Determinant_of_mapping_orthogonal);
//
//#pragma endregion
//	
//// for(float val = 0.1;val <= 6;val = val + 0.1){
//
//	for(int j = 0;j <  correctedImgHeight;j++){
//		for(int i = 0;i< correctedImgWidth;i++){
//            
//			cdy = j - center.y; 
//			cdx = i - center.x; 
//
//			nonlinear_mapping_orthogonal(cdx, cdy, xp, yp, foval, factor_c, factor_f);
//
//			yp = yp + center_fisheye.y;
//			xp = xp + center_fisheye.x;
//
//			// ������Χ�ĵ�����Ϊ����
//			if (xp < 0. || (int)xp >= fisheyeImgWidth || yp < 0. || (int)yp >= fisheyeImgHeight )
//			{ 
//					res = bgColor;
//			} 
//
//			/* ����ԭʼͼ��ľ������Ŀ�͸� */
//		    int Horizontalksize=0,Verticalksize=0;
//			computeConvArea(Horizontalksize, Verticalksize,  cdx, cdy, sigma, nonlinear_mapping_orthogonal, foval, factor_c, factor_f);
//
//		    Mat kernel( Verticalksize, Horizontalksize, CV_32FC1 );
//
//			//��ֵ��ʼ
//			//jzx �޸� ����Ҫ����0.5
//			//xp -= 0.5; yp -= 0.5;
//			xi = (int)floor( (double)xp ); 
//			yi = (int)floor( (double)yp );
//
//			/* ��ײ�ֵ */
//			/* zero order interpolation (pixel replication) */
//			if( order == 0 ){
//				
//				/*
//				fisheye_computeGaussKernel_withconv_withPreProc(kernel, sigma, 
//						xp,  yp, center_fisheye.x, center_fisheye.y, fisheyeImgWidth, fisheyeImgWidth,
//						cdx, cdy, foval, radius, factor_f, factor_c, inverse_nonlinear_mapping_orthogonal, S_xy);
//					
//				fisheye_perGaussianBlur2D_withconv(convResult, ref, fisheyeImgWidth*yi + xi, fisheyeImgWidth, fisheyeImgHeight,
//						kernel, Horizontalksize, Verticalksize);
//						 
//				fisheyeMIBlurImg.at<float>(yi,xi) = convResult;
//				*/
//				fisheyeMIBlurImg.at<float>(yi,xi) = fisheyeView[ yi * fisheyeImgWidth + xi ];
//				continue;
//			}
//
//			/* �߽ײ�ֵ */
//			ux = xp - (float)xi;
//			uy = yp - (float)yi;
//	
//			coefficientsOfHighOrderIntepolation(intervalStart,intervalEnd, cx, cy, ux, uy, paramOfBicubic, ak, order);
//
//	         res = 0.;
//			/* this test saves computation time */
//			if (xi > 0 && yi > 0 &&
//				xi + intervalStart >= 0 && xi + intervalEnd < fisheyeImgWidth && 
//				yi + intervalStart >= 0 && yi + intervalEnd < fisheyeImgHeight) {
//
//				adr = yi*fisheyeImgWidth + xi; 
//				
//				fisheye_computeGaussKernel_withconv_withPreProc(kernel, sigma,
//							xp,  yp, center_fisheye.x, center_fisheye.y, fisheyeImgWidth, fisheyeImgHeight,
//							cdx, cdy, foval, radius, factor_f, factor_c, inverse_nonlinear_mapping_orthogonal, S_xy);
//						 
//				for (dy = intervalStart;dy <= intervalEnd;dy++){
//					for (dx = intervalStart;dx <= intervalEnd;dx++){
//						 fisheye_perGaussianBlur2D_withconv(convResult, ref, adr+fisheyeImgWidth*dy+dx, fisheyeImgWidth, fisheyeImgHeight, kernel, Horizontalksize, Verticalksize);					 
//								res +=  cy[intervalEnd - dy] * cx[intervalEnd - dx] * convResult;
//					}
//				}
//			 } 
//			 else 
//			 {
//			 	for (dy = intervalStart;dy <= intervalEnd;dy++){
//			 		for (dx = intervalStart;dx <= intervalEnd;dx++){ 
//			 				res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * v( ref, xi + dx, yi + dy, bgColor, fisheyeImgWidth, fisheyeImgHeight); 
//			 		}
//			 	}
//			 } 
//
//			fisheyeMIBlurImg.at<float>(yi,xi) =  res;
//			//correctedImg.at<uchar>(j,i) = fisheyeImg.at<uchar>(yp,xp);
//		}
//	}
//
//	enhancementEffect_float(fisheyeMIBlurImg);
//
//	cv::imwrite(fileName, fisheyeMIBlurImg);
//
//}
//

// �������ɽ�����ͨ��˹������У��ͼ��
void generateCorrectedViewWithGaussianBlur(Mat &correctedViewImg, int correctedImgHeight,int correctedImgWidth, int radius, double camerFieldAngle, Mat fisheyeImg, double sigma, string fileName, float  factor = 1 /* ����ͼ��ķ�������*/){
  

	 int ksize = (int)(2.0 * 4.0 * sigma + 1.0);
	 cv::GaussianBlur(fisheyeImg,fisheyeImg, Size(ksize, ksize), sigma);

	Point2i center;
	center.y = (correctedImgHeight - 1) / 2;
	center.x = (correctedImgWidth - 1) / 2;
	
	Mat correctedImg(correctedImgHeight, correctedImgWidth, CV_32FC1);


	int fisheyeImgHeight = fisheyeImg.rows;
	int fisheyeImgWidth = fisheyeImg.cols;

	Point2i center_fisheye;
	center_fisheye.y = (fisheyeImgHeight - 1) / 2;
	center_fisheye.x = (fisheyeImgWidth - 1) / 2;
	
	double foval = 0.0;//����
	 // ��������ͶӰģ�ͼ���������� R_max = f * sin(theta_max), fovalΪ����
	foval = radius / sin(camerFieldAngle / 2);
 	
	vector<float> fisheyeView;

	matToVec_float(fisheyeImg, fisheyeView);

	/* INTERPOLATION */
	  int order = 1;
	float bgColor = 0.0;  // ��������            
	float paramOfBicubic = 0;  // float fperproj_p = 0; float *p = &fperproj_p;

	int  xi,yi;
	// ��ֵ�����������յ�
	int  intervalStart, intervalEnd;
	// adr Ҫ��ֵ�ĵ�, dxΪ��ֵ�������x�����ƫ����, dyΪ��ֵ�������y�����ƫ����
	int  adr,dx,dy; 
	float  res,ux,uy;
	float  cx[12],cy[12],ak[13];

	vector<float> ref, coeffs;

	uchar convResult = 0;
	generateInterpolatedView(ref, ak, fisheyeView, fisheyeImgWidth, fisheyeImgHeight, order);

	int cdx,cdy;
	double xp,yp;

	for(int j = 0;j <  correctedImgHeight;j++){
		for(int i = 0;i< correctedImgWidth;i++){
            
			cdy = j - center.y; 
			cdx = i - center.x; 

			nonlinear_mapping_orthogonal(cdx, cdy, xp, yp, foval);

			yp = yp + center_fisheye.y;
			xp = xp + center_fisheye.x;

			// ������Χ�ĵ�����Ϊ����
			if (xp < 0. || (int)xp >= fisheyeImgWidth || yp < 0. || (int)yp >= fisheyeImgHeight )
			{ 
					res = bgColor;
			} 

			//��ֵ��ʼ
			//jzx �޸� ����Ҫ����0.5
			//xp -= 0.5; yp -= 0.5;
			xi = (int)floor( (double)xp ); 
			yi = (int)floor( (double)yp );

			/* ��ײ�ֵ */
			/* zero order interpolation (pixel replication) */
			if( order == 0 ){
				
			    correctedImg.at<uchar>(j,i) = fisheyeView[ yi * fisheyeImgWidth + xi ];
				continue;
			}

			/* �߽ײ�ֵ */
			ux = xp - (float)xi;
			uy = yp - (float)yi;
	
			coefficientsOfHighOrderIntepolation(intervalStart,intervalEnd, cx, cy, ux, uy, paramOfBicubic, ak, order);

	         res = 0.;
			/* this test saves computation time */
			if (xi + intervalStart >= 0 && xi + intervalEnd < fisheyeImgWidth 
				&& yi + intervalStart >= 0 && yi + intervalEnd < fisheyeImgHeight) {

				adr = yi*fisheyeImgWidth + xi; 
	
				for(dy = intervalStart;dy <= intervalEnd;dy++){
					for(dx = intervalStart;dx <= intervalEnd;dx++){			
						 res +=  cy[intervalEnd - dy] * cx[intervalEnd - dx] * ref[adr + fisheyeImgWidth * dy + dx];
					}
				}
				
			 } 
			 else 
			 {
			 	for (dy = intervalStart;dy <= intervalEnd;dy++){
			 		for (dx = intervalStart;dx <= intervalEnd;dx++){ 
			 				res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * v( ref, xi + dx, yi + dy, bgColor, fisheyeImgWidth, fisheyeImgHeight); 
			 		}
			 	}
			 } 

			correctedImg.at<float>(j,i) =  res;
			//correctedImg.at<uchar>(j,i) = fisheyeImg.at<uchar>(yp,xp);
		}
	}

	enhancementEffect_float(correctedImg);
	cv::imwrite(fileName, correctedImg);

}

// �������ɽ���MI������У��ͼ��
void generateCorrectedViewWithMIConv(Mat &correctedViewImg, int correctedImgHeight,int correctedImgWidth, int radius, double camerFieldAngle, Mat fisheyeImg, double sigma, string fileName, float  factor_f = 1 /* ����ͼ��ķ�������*/, float  factor_c = 1 /* У��ͼ��ķ�������*/){
  

	correctedImgHeight = correctedImgHeight / factor_c;
	correctedImgWidth = correctedImgWidth / factor_c;

	Point2i center;
	center.y = (correctedImgHeight - 1) / 2;
	center.x = (correctedImgWidth - 1) / 2;
	
	Mat correctedImg(correctedImgHeight, correctedImgWidth, CV_32FC1);

	int fisheyeImgHeight = fisheyeImg.rows;
	int fisheyeImgWidth = fisheyeImg.cols;

	Point2i center_fisheye;
	center_fisheye.y = (fisheyeImgHeight - 1) / 2;
	center_fisheye.x = (fisheyeImgWidth - 1) / 2;
	
	double foval = 0.0;//����
	 // ��������ͶӰģ�ͼ���������� R_max = f * sin(theta_max), fovalΪ����
	foval = radius / sin(camerFieldAngle / 2);
 	
	vector<float> fisheyeView;

	matToVec_float(fisheyeImg, fisheyeView);

	/* INTERPOLATION */
	  int order = 1;
	float bgColor = 0.0;  // ��������            
	float paramOfBicubic = 0;  // float fperproj_p = 0; float *p = &fperproj_p;

	int  xi,yi;
	// ��ֵ�����������յ�
	int  intervalStart, intervalEnd;
	// adr Ҫ��ֵ�ĵ�, dxΪ��ֵ�������x�����ƫ����, dyΪ��ֵ�������y�����ƫ����
	int  adr,dx,dy; 
	float  res,ux,uy;
	float  cx[12],cy[12],ak[13];

	vector<float> ref, coeffs;

	uchar convResult = 0;
	generateInterpolatedView(ref, ak, fisheyeView, fisheyeImgWidth, fisheyeImgHeight, order);

	int cdx,cdy;
	double xp,yp;

	

#pragma region ����ԭͼ����ÿ�����Ȩ��

	// Ϊ��������ʼ��ֵ����ֹ����ʱ����
   vector<float> S_xy(fisheyeImgWidth * fisheyeImgHeight, 0);
   //����ԭͼ����ÿ��������Ȩ��,ÿ��ģ��ͼ���Ӧһ�����Ȩ��
	areaWeight_orthogonal(S_xy,fisheyeImgWidth, fisheyeImgHeight, center_fisheye.x, center_fisheye.y, foval, radius, factor_f, factor_c,
		inverse_nonlinear_mapping_orthogonal, abs_jacobian_Determinant_of_mapping_orthogonal);

#pragma endregion
	
// for(float val = 0.1;val <= 6;val = val + 0.1){

	for(int j = 0;j <  correctedImgHeight;j++){
		for(int i = 0;i< correctedImgWidth;i++){
            
			cdy = (j - center.y); 
			cdx = (i - center.x); 

			nonlinear_mapping_orthogonal(cdx, cdy, xp, yp, foval, factor_c, factor_f);

			yp = (yp  + center_fisheye.y) ;
			xp = (xp  + center_fisheye.x) ;

			// ������Χ�ĵ�����Ϊ����
			if (xp < 0. || (int)xp >= fisheyeImgWidth || yp < 0. || (int)yp >= fisheyeImgHeight )
			{ 
					res = bgColor;
			} 

			/* ����ԭʼͼ��ľ������Ŀ�͸� */
		    int Horizontalksize=0,Verticalksize=0;
			computeConvArea(Horizontalksize, Verticalksize,  cdx, cdy, sigma, nonlinear_mapping_orthogonal, foval, factor_c, factor_f);

		    Mat kernel( Verticalksize, Horizontalksize, CV_32FC1 );

			//��ֵ��ʼ
			//jzx �޸� ����Ҫ����0.5
			//xp -= 0.5; yp -= 0.5;
			xi = (int)floor( (double)xp ); 
			yi = (int)floor( (double)yp );

			/* ��ײ�ֵ */
			/* zero order interpolation (pixel replication) */
			if( order == 0 ){
				
				
				fisheye_computeGaussKernel_withconv_withPreProc(kernel, sigma, 
						xp,  yp, center_fisheye.x, center_fisheye.y, fisheyeImgWidth, fisheyeImgWidth,
						cdx, cdy, foval, radius, factor_f, factor_c, inverse_nonlinear_mapping_orthogonal, S_xy);
				
				

				fisheye_perGaussianBlur2D_withconv(convResult, ref, fisheyeImgWidth*yi + xi, fisheyeImgWidth, fisheyeImgHeight,
						kernel, Horizontalksize, Verticalksize);
						 
				correctedImg.at<float>(j,i) = convResult;
				
				//correctedImg.at<float>(j,i) = fisheyeView[ yi * fisheyeImgWidth + xi ];
				continue;
			}

			/* �߽ײ�ֵ */
			ux = xp - (float)xi;
			uy = yp - (float)yi;
	
			coefficientsOfHighOrderIntepolation(intervalStart,intervalEnd, cx, cy, ux, uy, paramOfBicubic, ak, order);

	         res = 0.;
			/* this test saves computation time */
			if (xi > 0 && yi > 0  &&
				xi + intervalStart >= 0 && xi + intervalEnd < fisheyeImgWidth && 
				yi + intervalStart >= 0 && yi + intervalEnd < fisheyeImgHeight) {

				adr = yi*fisheyeImgWidth + xi; 
				
				fisheye_computeGaussKernel_withconv_withPreProc(kernel, sigma,
							xp,  yp, center_fisheye.x, center_fisheye.y, fisheyeImgWidth, fisheyeImgHeight,
							cdx, cdy, foval, radius, factor_f, factor_c, inverse_nonlinear_mapping_orthogonal, S_xy);
						 

				for (dy = intervalStart;dy <= intervalEnd;dy++){
					for (dx = intervalStart;dx <= intervalEnd;dx++){
						 fisheye_perGaussianBlur2D_withconv(convResult, ref, adr+fisheyeImgWidth*dy+dx, fisheyeImgWidth, fisheyeImgHeight, kernel, Horizontalksize, Verticalksize);					 
						  res +=  cy[intervalEnd - dy] * cx[intervalEnd - dx] * convResult;
						  // res +=  cy[intervalEnd - dy] * cx[intervalEnd - dx] * ref[ adr+fisheyeImgWidth*dy+dx ];



					}
				}
			 } 
			 else 
			 {
			 	for (dy = intervalStart;dy <= intervalEnd;dy++){
			 		for (dx = intervalStart;dx <= intervalEnd;dx++){ 
			 				res += cy[intervalEnd - dy] * cx[intervalEnd - dx] * v( ref, xi + dx, yi + dy, bgColor, fisheyeImgWidth, fisheyeImgHeight); 
			 		}
			 	}
			 } 

			correctedImg.at<float>(j,i) =  res;
			//correctedImg.at<uchar>(j,i) = fisheyeImg.at<uchar>(yp,xp);
		}
	}

	enhancementEffect_float(correctedImg);
	correctedViewImg = correctedImg;
	cv::imwrite(fileName, correctedImg);
	
}






int main(int argc, char** argv){


		
	// �������ڱ������ݵ�Ŀ¼

	string dirPV = "ProblemVisulization";
	CreateDirectory(dirPV.data(),NULL);

	double sigma = 0.14;

	int ImgW = 256 ;//128;//64;
	int ImgH = 256 ;//128;//64;


	vector<Mat>  FrontViewArry;
	FrontViewArry.push_back( Mat( ImgH, ImgW, CV_32FC1, Scalar(0) ) );
	FrontViewArry.push_back( Mat( ImgH, ImgW, CV_32FC1, Scalar(0) ) );

#pragma region ��������ͼ��

	for(int yCount=0;yCount<ImgH;yCount++)
	{
		for(int xCount=0;xCount<ImgW;xCount++)
		{
			FrontViewArry[0].at<float>(yCount, xCount) +=cos(((float)(2*PI*yCount))/50)*cos(((float)(2*PI*xCount))/50)+2*cos(((float)(9*2*PI*yCount))/20)*cos(((float)(9*2*PI*xCount))/20);
			FrontViewArry[1].at<float>(yCount, xCount) +=cos(((float)(2*PI*yCount))/50)*cos(((float)(2*PI*xCount))/50)+2*cos(((float)(1*2*PI*yCount))/20)*cos(((float)(1*2*PI*xCount))/20);
		}
	}

	float MinV0 = 0;
	float MinV1 = 0;
	for(int yCount=0;yCount<ImgH;yCount++)
	{
		for(int xCount=0;xCount<ImgW;xCount++)
		{
			if(FrontViewArry[0].at<float>(yCount, xCount) < MinV0)
			{
				MinV0 = FrontViewArry[0].at<float>(yCount, xCount);
			}
			if(FrontViewArry[1].at<float>(yCount, xCount) < MinV1)
			{
				MinV1 = FrontViewArry[1].at<float>(yCount, xCount);
			}
		}
	}
	//debug
	//printf("MinV = %f\n",MinV);
	//
	for(int yCount=0;yCount<ImgH;yCount++)
	{
		for(int xCount=0;xCount<ImgW;xCount++)
		{
			FrontViewArry[0].at<float>(yCount, xCount) += abs(MinV0);
			FrontViewArry[1].at<float>(yCount, xCount) += abs(MinV1);
		}
	}
	float MaxV0 = 0;
	float MaxV1 = 0;
	for(int yCount=0;yCount<ImgH;yCount++)
	{
		for(int xCount=0;xCount<ImgW;xCount++)
		{
			if(FrontViewArry[0].at<float>(yCount, xCount) > MaxV0)
			{
				MaxV0 = FrontViewArry[0].at<float>(yCount, xCount);
			}
			if(FrontViewArry[1].at<float>(yCount, xCount) > MaxV1)
			{
				MaxV1 = FrontViewArry[1].at<float>(yCount, xCount);
			}
		}
	}

	if(MaxV0>0)
	{
		for(int yCount=0;yCount<ImgH;yCount++)
		{
			for(int xCount=0;xCount<ImgW;xCount++)
			{
				FrontViewArry[0].at<float>(yCount, xCount) = 255*FrontViewArry[0].at<float>(yCount, xCount)/ MaxV0;
			}
		}
	}

	if(MaxV1>0)
	{
		for(int yCount=0;yCount<ImgH;yCount++)
		{
			for(int xCount=0;xCount<ImgW;xCount++)
			{
				FrontViewArry[1].at<float>(yCount, xCount) = 255*FrontViewArry[1].at<float>(yCount, xCount) / MaxV1;
			}
		}
	}

#pragma endregion 


   string fileName;
   char TempStr[150];
       
   int mImgH =ImgH / 4, mImgW = ImgW / 4;


vector<Mat>  correctedGaussBlurViewArry(2);
	//correctedGaussBlurViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );
	//correctedGaussBlurViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );

vector<Mat>  correctedMIBlurViewArry(2);
	//correctedMIBlurViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );
	//correctedMIBlurViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );

vector<Mat>  correctedViewArry(2);
	//correctedViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );
	//correctedViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );

vector<Mat>  correctedBehindBlurViewArry(2);
	//correctedBehindBlurViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );
	//correctedBehindBlurViewArry.push_back( Mat( mImgH, mImgW, CV_32FC1, Scalar(0) ) );




for(int iCount=0;iCount<2;iCount++)
{
	 //Size dsize = Size(ImgH * 0.1, ImgW * 0.1);  
  //   Mat imagedst = Mat(dsize, CV_32S);  
  //   resize(FrontViewArry[iCount], imagedst, dsize);
	 //sprintf(TempStr,"ProblemVisulization/minFrontView%d.png",iCount);
	 //fileName = string(TempStr);
	 //cv::imwrite(fileName, imagedst);

	 sprintf(TempStr,"ProblemVisulization/FrontView%d.png",iCount);
	 fileName = string(TempStr);
	 cv::imwrite(fileName, FrontViewArry[iCount]);

#pragma region ���ɸ�˹ģ��ͼ�񣨶�ģ������Ч��������ǿ��


	 Mat blurViewImg;
	 sprintf(TempStr,"ProblemVisulization/FrontViewBlur%d.png",iCount);
	 fileName = string(TempStr);
	 generateGaussianBlurView_float(FrontViewArry[iCount], blurViewImg, sigma, fileName);

#pragma endregion

	 //resize(blurViewImg, imagedst, dsize);
	 //sprintf(TempStr,"ProblemVisulization/min1FrontView%d.png",iCount);
	 //fileName = string(TempStr);
	 //cv::imwrite(fileName, imagedst);
}

   


int radius = 128;  

for( ; radius <= 256 ; radius++ ){

 int iCount=0;
 
for(;iCount<2;iCount++)
{

       sprintf(TempStr,"ProblemVisulization/fisheye%d",iCount);
	   fileName = string(TempStr);
	   CreateDirectory(fileName.data(),NULL);

	   Mat  frontViewImg =  FrontViewArry[iCount];

#pragma region ��������ͼ��

	   double factor_f = 4; 
	   double factor_c = 4;

	   double camerFieldAngle = PI;
	   Mat fisheyeImg;

	   sprintf(TempStr,"ProblemVisulization/fisheye%d/fisheyeImg%d_%d.png",iCount,iCount, radius);
	   fileName = string(TempStr);
	   generateFisheyeView(fisheyeImg, radius, camerFieldAngle, frontViewImg, fileName, factor_f);
	   // generateSpherizeView(fisheyeImg, radius, camerFieldAngle, frontViewImg, fileName); 
	   // cv::imwrite(fileName, FrontViewArry[iCount]);
	   
	   // ȥ��ͼ��������С�����ص�
	   for(int j = 0;j < fisheyeImg.rows;j++){
		   for(int i = 0;i<fisheyeImg.cols;i++){
		       if( !_finite( fisheyeImg.at<float>(j,i) ) )
			       fisheyeImg.at<float>(j,i) = 0;
		   }
	   }

#pragma endregion   

#pragma region ������ͼ����и�˹ģ������

	   Mat fisheyeGaussianBlurImg;
	   sprintf(TempStr,"ProblemVisulization/fisheye%d/fisheyeImg_GaussianBlur%d_%d.png",iCount,iCount,radius);
	   fileName = string(TempStr);
	   generateGaussianBlurView_float(fisheyeImg, fisheyeGaussianBlurImg, sigma, fileName);
	
#pragma endregion 

#pragma region �Խ��й���˹ģ����ͼ�����У������
	   
	   Mat correctedImg_GaussianBlur;
	   sprintf(TempStr,"ProblemVisulization/fisheye%d/correctedImg_GaussianBlur%d_%d.png",iCount,iCount,radius);
	   fileName = string(TempStr);
	   //generateCorrectedViewWithGaussianBlur(correctedImg_GaussianBlur, ImgH * 0.5, ImgW * 0.5, radius, camerFieldAngle, fisheyeImg, sigma, fileName, factor);
	   generateCorrectedView(correctedImg_GaussianBlur, ImgH, ImgW, radius, camerFieldAngle, fisheyeGaussianBlurImg, fileName, factor_f,factor_c);
	  // cv::imwrite(fileName, FrontViewArry[iCount]);


	   correctedGaussBlurViewArry[iCount] = correctedImg_GaussianBlur;


#pragma endregion 

//#pragma region ������ͼ�����MIģ������
//	   
//	   Mat fisheyeMIBlurImg;
//	   sprintf(TempStr,"ProblemVisulization/fisheye%d/fisheyeImg_MIBlur%d.png",iCount,iCount);
//	   fileName = string(TempStr);
//	  // cv::imwrite(fileName, FrontViewArry[iCount]);
//	   generatefisheyeViewWithMIConv(fisheyeMIBlurImg, ImgH, ImgW, radius, camerFieldAngle, fisheyeImg, sigma, fileName, factor_f, factor_c);
//
//#pragma endregion

#pragma region ������ͼ����й�MIģ�������ͼ�����У������

	   Mat correctedImg_MIBlur;   
	   sprintf(TempStr,"ProblemVisulization/fisheye%d/correctedImg_MIBlur%d_%d.png",iCount,iCount,radius);
	   fileName = string(TempStr);
	   generateCorrectedViewWithMIConv(correctedImg_MIBlur, ImgH, ImgW, radius, camerFieldAngle, fisheyeImg, sigma, fileName, factor_f, factor_c);
	   
	   correctedMIBlurViewArry[iCount] = correctedImg_MIBlur; 

#pragma endregion 

#pragma region ������ͼ�񲻽���ģ��������ֱ�ӽ���У��

	   Mat correctedViewImg;
       sprintf(TempStr,"ProblemVisulization/fisheye%d/correctedImg_withoutBlur%d_%d.png",iCount,iCount,radius);
	   fileName = string(TempStr);
	   generateCorrectedView(correctedViewImg, ImgH, ImgW, radius, camerFieldAngle, fisheyeImg, fileName, factor_f,factor_c);
	   // cv::imwrite(fileName, FrontViewArry[iCount]);

	   correctedViewArry[iCount] = correctedViewImg;

#pragma endregion 

#pragma region ����һ��У������ͼ�����ģ������

	   Mat correctedViewImg_GaussianBlur;
	   sprintf(TempStr,"ProblemVisulization/fisheye%d/correctedImg_behindGaussianBlur%d_%d.png",iCount,iCount,radius);
	   fileName = string(TempStr);
	   // cv::imwrite(fileName, FrontViewArry[iCount]);
       generateGaussianBlurView_float(correctedViewImg, correctedViewImg_GaussianBlur, sigma, fileName);

	   correctedBehindBlurViewArry[iCount] = correctedViewImg_GaussianBlur;
#pragma endregion 

}



	
#pragma region �����Զ���

    float GBAve[] = {0,0};
	float MIAve[] = {0,0};
	float CAve[] = {0,0};
	float BBAve[] = {0,0};

	float GBMod[] = {0,0};
	float MIMod[] = {0,0};
	float CMod[] = {0,0};
	float BBMod[] = {0,0};

	int vecLength = mImgH*mImgW;



correctedGaussBlurViewArry[0];
correctedMIBlurViewArry[0];
correctedViewArry[0];
correctedBehindBlurViewArry[0];




 for(int ind = 0;ind < 2;ind++){
	for(int h = 0;h < mImgH;h++){
		for(int w = 0;w <  mImgW;w++){
		     GBAve[ind] += correctedGaussBlurViewArry[ind].at<float>(h, w)/(vecLength);
	         MIAve[ind] += correctedMIBlurViewArry[ind].at<float>(h, w)/(vecLength);
	          CAve[ind] += correctedViewArry[ind].at<float>(h, w)/(vecLength);
	         BBAve[ind] += correctedBehindBlurViewArry[ind].at<float>(h, w)/(vecLength);
		}
	}
 }

 
 for(int ind = 0;ind < 2;ind++){
	for(int h = 0;h < mImgH;h++){
		for(int w = 0;w <  mImgW;w++){
		     correctedGaussBlurViewArry[ind].at<float>(h, w) = correctedGaussBlurViewArry[ind].at<float>(h, w) - GBAve[ind];
	         correctedMIBlurViewArry[ind].at<float>(h, w) =correctedMIBlurViewArry[ind].at<float>(h, w)- MIAve[ind];
	         correctedViewArry[ind].at<float>(h, w) = correctedViewArry[ind].at<float>(h, w)- CAve[ind];
	         correctedBehindBlurViewArry[ind].at<float>(h, w)=correctedBehindBlurViewArry[ind].at<float>(h, w)- BBAve[ind];
		}
	}
 }

  for(int ind = 0;ind < 2;ind++){
	for(int h = 0;h < mImgH;h++){
		for(int w = 0;w <  mImgW;w++){
		    GBMod[ind] +=  correctedGaussBlurViewArry[ind].at<float>(h, w) * correctedGaussBlurViewArry[ind].at<float>(h, w);
	         MIMod[ind] += correctedMIBlurViewArry[ind].at<float>(h, w) * correctedMIBlurViewArry[ind].at<float>(h, w);
	        CMod[ind] += correctedViewArry[ind].at<float>(h, w) * correctedViewArry[ind].at<float>(h, w);
	        BBMod[ind] += correctedBehindBlurViewArry[ind].at<float>(h, w)* correctedBehindBlurViewArry[ind].at<float>(h, w);
		}
	}
 }


  for(int ind = 0;ind < 2;ind++){
	
		 GBMod[ind] = sqrt(GBMod[ind]);
	     MIMod[ind] = sqrt(MIMod[ind]);
	      CMod[ind] = sqrt(CMod[ind]);
	     BBMod[ind] = sqrt(BBMod[ind]);	
 }


  	float Similarity_GB = 0;
	float Similarity_MI = 0;
	float Similarity_C = 0;
	float Similarity_BB = 0;


	for(int h = 0;h < mImgH;h++){
		for(int w = 0;w <  mImgW;w++){
		   Similarity_GB +=  correctedGaussBlurViewArry[0].at<float>(h, w) * correctedGaussBlurViewArry[1].at<float>(h, w) / (GBMod[0] * GBMod[1]);
	       Similarity_MI += correctedMIBlurViewArry[0].at<float>(h, w) * correctedMIBlurViewArry[1].at<float>(h, w) /(MIMod[0] * MIMod[1]);
	       Similarity_C += correctedViewArry[0].at<float>(h, w) * correctedViewArry[1].at<float>(h, w) /(CMod[0] * CMod[1]);
	       Similarity_BB += correctedBehindBlurViewArry[0].at<float>(h, w)* correctedBehindBlurViewArry[1].at<float>(h, w) /(BBMod[0] * BBMod[1]);
		}
	}


	float AveError_GB = 0;
	float AveError_MI = 0;
	float AveError_C = 0;
	float AveError_BB = 0;

		for(int h = 0;h < mImgH;h++){
		for(int w = 0;w <  mImgW;w++){
		  AveError_GB += abs(correctedGaussBlurViewArry[0].at<float>(h, w) - correctedGaussBlurViewArry[1].at<float>(h, w));
	      AveError_MI += abs(correctedMIBlurViewArry[0].at<float>(h, w) - correctedMIBlurViewArry[1].at<float>(h, w));
	      AveError_C += abs(correctedViewArry[0].at<float>(h, w) - correctedViewArry[1].at<float>(h, w));
	      AveError_BB +=abs(correctedBehindBlurViewArry[0].at<float>(h, w) - correctedBehindBlurViewArry[1].at<float>(h, w));
		}
	}

    
    AveError_GB = AveError_GB / vecLength;
	AveError_MI = AveError_MI /vecLength;
	 AveError_C =AveError_C /vecLength;
    AveError_BB = AveError_BB / vecLength;



	sprintf(TempStr,"ProblemVisulization/Similarity.txt");
	String str1 = string(TempStr);
	std::ofstream similarityFile(str1.data(),ios::app);	
	//similarityFile<<"�뾶:"<<radius<<endl;
	//similarityFile<<"�̶�������ƶ�:"<<Similarity_GB<<" �̶��������:"<<AveError_GB<<endl;
	//similarityFile<<"MI������ƶ�:"<<Similarity_MI<<" MI�������:"<<AveError_MI<<endl;
	//similarityFile<<"û����������ƶ�:"<<Similarity_C<<" �̶��������:"<<AveError_C<<endl;
	//similarityFile<<"У���������������ƶ�:"<<Similarity_BB<<" MI�������:"<<AveError_BB<<endl;

	similarityFile<< radius<<"\t"
		<<Similarity_GB<<"\t"<<Similarity_MI<<"\t"<<Similarity_C<<"\t"<<Similarity_BB<<"\t"
		<<AveError_GB<<"\t"<<AveError_MI<<"\t"<<AveError_C<<"\t"<<AveError_BB<<endl;



#pragma endregion
















 }

   return 0;

}