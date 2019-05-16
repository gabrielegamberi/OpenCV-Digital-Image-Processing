#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <list>
/*HARRIS CON L'UTILIZZO DELLA HARRIS RESPONSE (FUNZIONANTE)*/
using namespace cv;
using namespace std;

#define SIZE 5
#define PAD floor(SIZE/2)
#define SOBEL_SIZE 3
#define SOBEL_PAD floor(SOBEL_SIZE/2)

Mat sourceMatrix, grayMatrix;
Mat covMatrix, harrisResponse;

const char* harrisWindow = "My Harris corner detector";

Mat kernel(SIZE,SIZE,CV_32FC1,Scalar(0));

Mat horSobel = (Mat_<char>(SOBEL_SIZE,SOBEL_SIZE)<<-1,-2,-1,
									   				 0, 0, 0,
									   				 1, 2, 1
);
Mat vertSobel = (Mat_<char>(SOBEL_SIZE,SOBEL_SIZE)<<-1, 0, 1,
									    			-2, 0, 2,
									    			-1, 0, 1
);

class Pixel{
	private:
		int row;
		int col;
		float response;
	public:
		Pixel(int newRow, int newCol, int newReponse): row(newRow), col(newCol), response(newReponse){}
		bool operator<(const Pixel &anotherPixel)const{
			return (anotherPixel.response<response);
		}
		inline float getResponse(){return this->response;}
		inline int getRow(){return this->row;}
		inline int getCol(){return this->col;}
};

void displayHistogram(const Mat& image){
    int N = image.rows;
    int M = image.cols;
	int L = 256;
    vector<int> histogram(L,0);
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            histogram.at(image.at<uchar>(i,j))++;
    //FIND THE MAX VALUE OF THE BINS
    int max = histogram.at(0);
    for(int i = 1; i < L; i++)
        if(max < histogram.at(i))
            max = histogram.at(i);
    int hist_width = 1024;
    int hist_height = 400;
    int bin_widht = round((float)hist_width/L);
    Mat histogramImage(hist_height+24, hist_width+48, CV_8UC1, Scalar(255, 255, 255));
    for(int i=0; i<L; i++)
        histogram.at(i) = ((double)histogram.at(i)/max)*hist_height;
    //DRAW THE LINE
    for(int i = 0; i < L; i++)
      //line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
        line(histogramImage, Point(bin_widht*(i)+24, hist_height), Point(bin_widht*(i)+24, hist_height - histogram.at(i)),Scalar(0,0,0), 2, 8, 0);
    imshow("histogram", histogramImage);
    
}

void buildGaussianFilter(){
	float variance = 4;
	int rowShift, colShift, B;
	float minTerm;
	for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			kernel.at<float>(i,j) = exp(-(pow(abs(i-PAD),2)+pow(abs(j-PAD),2))/(2*variance));
	minTerm = kernel.at<float>(kernel.rows-1,kernel.cols-1);

	for(int i=0; i<kernel.rows; i++){
		for(int j=0; j<kernel.cols; j++){
			B = round(kernel.at<float>(i,j)/minTerm);
			kernel.at<float>(i,j) = B;	
		}
	}
}

void gaussianFilter(Mat& inoutPaddedMatrix){
	int denom=0;
	Mat temp = inoutPaddedMatrix.clone();

	for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			denom+=kernel.at<uchar>(i,j);

	int maskedValue[SIZE*SIZE];
	for(int row=PAD; row<temp.rows-PAD; row++){
		for(int col=PAD; col<temp.cols-PAD; col++){
			int mask_index = 0;
			for(int i=-PAD; i<=PAD; i++)
				for(int j=-PAD; j<=PAD; j++)
					maskedValue[mask_index++] = temp.at<uchar>(row+i,col+j)*kernel.at<float>(i+PAD, j+PAD);
			int sum=0;
			for(int i=0; i<SIZE*SIZE; i++)
				sum+=maskedValue[i];
			sum=floor(sum/denom);
			inoutPaddedMatrix.at<uchar>(row,col) = sum;
		}
	}
}

void sobel(const Mat& inputMatrix, Mat& sobelGx, Mat& sobelGy){
	int yGrad, xGrad;
	int N = inputMatrix.rows;
	int M = inputMatrix.cols;
	sobelGx = Mat(inputMatrix.size(), CV_8U);	
	sobelGy = Mat(inputMatrix.size(), CV_8U);
	
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			yGrad = xGrad = 0;
			for(int i=-SOBEL_PAD; i<=SOBEL_PAD; i++){
				for(int j=-SOBEL_PAD; j<=SOBEL_PAD; j++){
					yGrad+=(int)(inputMatrix.at<uchar>(row+i,col+j)*vertSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
					xGrad+=(int)(inputMatrix.at<uchar>(row+i,col+j)*horSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
				}
			}
			sobelGx.at<uchar>(row,col) = saturate_cast<uchar>(abs(xGrad));
			sobelGy.at<uchar>(row,col) = saturate_cast<uchar>(abs(yGrad));
		}
	}
	namedWindow("X",WINDOW_AUTOSIZE);
	imshow("X", sobelGx);
	namedWindow("Y",WINDOW_AUTOSIZE);
	imshow("Y", sobelGy);
	
}

float calcResponse(int row, int col, const Mat& gx, const Mat& gy, int windowSize){
	int pad = floor(windowSize/2);
	float k = 0.04f;
	Mat covarianceMat = Mat::zeros(2,2,CV_32F);
	for(int i=-pad; i<=pad; i++){
		for(int j=-pad; j<=pad;j ++){
			covarianceMat.at<float>(0,0)+=(float)pow(gx.at<uchar>(row+i,col+j),2);
			covarianceMat.at<float>(0,1)+=(float)gx.at<uchar>(row+i,col+j)*gy.at<uchar>(row+i,col+j);
			covarianceMat.at<float>(1,0)+=(float)gx.at<uchar>(row+i,col+j)*gy.at<uchar>(row+i,col+j);
			covarianceMat.at<float>(1,1)+=(float)pow(gy.at<uchar>(row+i,col+j),2);
			
		}
	}
	float det = covarianceMat.at<float>(0,0)*covarianceMat.at<float>(1,1)-(covarianceMat.at<float>(0,1)*covarianceMat.at<float>(1,0));
	return (det-k*(pow(trace(covarianceMat).val[0],2))); 			
}
void suppressCorners(list<Pixel> responses, Mat &inoutMatrix){
	responses.sort();
	list<Pixel>::iterator it,it2;
	int neighborhood = 15;
	int pad = floor(neighborhood/2);
	float limit = sqrt(pow(pad,2)+pow(pad,2));
	for(it=responses.begin();it!=responses.end();){
		Pixel currentCorner = *it;
		it2 = it;
		advance(it2,1);
		for(; it2!=responses.end();){
			Pixel neighbor = *it2;
			float pixelDistance = sqrt(pow(neighbor.getRow()-currentCorner.getRow(),2)+pow(neighbor.getCol()-currentCorner.getCol(),2));
			if(pixelDistance<=limit){
				it2 = responses.erase(it2);
			}else{
				++it2;
			}
		}
		++it;
	}
	for(it=responses.begin(); it!=responses.end(); it++)
		circle(inoutMatrix, Point(it->getCol(),it->getRow()), pad, Scalar(150), 1, 4);
}

void detectHarrisCorners(){
	Mat paddedMatrix = Mat::zeros(grayMatrix.rows+2*PAD, grayMatrix.cols+2*PAD, grayMatrix.type());
	Mat sobelGx, sobelGy;

	for(int i=PAD; i<paddedMatrix.rows-PAD; i++)
		for(int j=PAD; j<paddedMatrix.cols-PAD; j++)
			paddedMatrix.at<uchar>(i,j) = grayMatrix.at<uchar>(i-PAD, j-PAD);

	buildGaussianFilter();
	sobel(paddedMatrix, sobelGx, sobelGy);
	
    gaussianFilter(sobelGx);
    gaussianFilter(sobelGy);

	Mat beforeMatrix = paddedMatrix.clone();
	harrisResponse = Mat(paddedMatrix.size(), CV_32FC1);
	
	list<Pixel> responses;
	int neighborhood = 5;
	for(int row=PAD; row<paddedMatrix.rows-PAD; row++){
		for(int col=PAD; col<paddedMatrix.cols-PAD; col++){
			harrisResponse.at<float>(row,col) = calcResponse(row,col,sobelGx,sobelGy,neighborhood);
			Pixel pixel(row,col,harrisResponse.at<float>(row,col));
			if(pixel.getResponse()>100.0f){
				circle(beforeMatrix, Point(col,row), 5, Scalar(150), 1, 4);
				responses.push_front(pixel);
			}
		}
	}

	namedWindow("before",WINDOW_AUTOSIZE);
	imshow("before", beforeMatrix);
	
	suppressCorners(responses, paddedMatrix);
	
	namedWindow("after",WINDOW_AUTOSIZE);
	imshow("after", paddedMatrix);
}

int main( int argc, char** argv ){
    const char* fileName = (argc>=2)? argv[1]:"./scacchi.jpg";
	sourceMatrix = imread(fileName, IMREAD_COLOR);
    
	if (sourceMatrix.empty()){
        cout<<"--- EMPTY IMAGE ---"<<endl;
        exit(EXIT_FAILURE);
    }

    cvtColor(sourceMatrix, grayMatrix, COLOR_BGR2GRAY );
    detectHarrisCorners();

    waitKey(0);
    return 0;
}
/*#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define GAUSS_LEN 5 //DIMENSIONI KERNEL PER SMOOTHING GAUSSIANO
#define PAD floor(GAUSS_LEN/2)

Mat horizSobel = (Mat_<char>(3,3) <<-1, -2, -1,
                                    0,  0,  0,
                                    1,  2,  1);
Mat vertSobel = (Mat_<char>(3,3) <<-1, 0, 1,
                                    -2, 0, 2,
                                    -1, 0, 1);
class Pixel{
	private:
		int row;
		int col;
		float response;
	public:
		Pixel(int newRow, int newCol, int newReponse): row(newRow), col(newCol), response(newReponse){}
		bool operator<(const Pixel &anotherPixel)const{
			return (anotherPixel.response<response);
		}
		inline float getResponse(){return this->response;}
		inline int getRow(){return this->row;}
		inline int getCol(){return this->col;}
};

void sobel(const Mat& inputImage, Mat& gx, Mat& gy){
    int N = inputImage.rows;
    int M = inputImage.cols;
    int xGradient, yGradient;
    for(int row=PAD; row<N-PAD; row++){
        for(int col=PAD; col<M-PAD; col++){
            xGradient = 0, yGradient = 0;
            for(int k=-1; k<=1; k++){
                for(int l=-1; l<=1; l++){
                    xGradient += inputImage.at<uchar>(row+k,col+l)*horizSobel.at<char>(k+1,l+1);
                    yGradient += inputImage.at<uchar>(row+k,col+l)*vertSobel.at<char>(k+1,l+1);
                }
            }
            gx.at<uchar>(row,col) = abs(xGradient);
            gy.at<uchar>(row,col) = abs(yGradient);
        }
    }
    // imshow("gx", gx);
    // imshow("gy", gy);
}

void gaussianFilter(Mat& inoutMatrix){
    Mat temp = inoutMatrix.clone();
    Mat kernel = Mat::zeros(GAUSS_LEN, GAUSS_LEN, CV_32FC1);
    int N = inoutMatrix.rows;
    int M = inoutMatrix.cols;
    short variance = 3;
    //POPOLO IL KERNEL GAUSSIANO
    for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			kernel.at<float>(i,j) = exp(-(pow(abs(i-PAD),2)+pow(abs(j-PAD),2))/(2*variance));

    float minTerm = kernel.at<float>(kernel.rows-1,kernel.cols-1);

    for(int i=0; i<kernel.rows; i++){
		for(int j=0; j<kernel.cols; j++){
			int B = round(kernel.at<float>(i,j)/minTerm);
			kernel.at<float>(i,j) = B;	
		}
	}

    float denom = 0;
    for(int i=0; i<GAUSS_LEN; i++)
        for(int j=0; j<GAUSS_LEN; j++)
            denom += kernel.at<float>(i,j);

    //APPLICO LO SMOOTHING GAUSSIANO
    int count;
    for(int row=PAD; row<N-PAD; row++){
        for(int col=PAD; col<M-PAD; col++){
            count = 0;
            for(int k=-PAD; k<=PAD; k++){
                for(int l=-PAD; l<=PAD; l++)
                     count += (int)inoutMatrix.at<uchar>(row+k,col+l)*kernel.at<float>(k+PAD,l+PAD);
                inoutMatrix.at<uchar>(row,col) = round(count/denom); //normalizzo
            }
        }
    }
    //imshow("smoothed image", outputImage);
}

void createPaddedMatrix(const Mat& rawImage, Mat& outputImage){
    int N = rawImage.rows;
    int M = rawImage.cols;
    outputImage = Mat::zeros(N+2*PAD,M+2*PAD, rawImage.type());  
    for(int row=PAD; row<N-PAD; row++)
        for(int col=PAD; col<M-PAD; col++)
            outputImage.at<uchar>(row,col) = rawImage.at<uchar>(row-PAD,col-PAD);
    //imshow("framed image", image);
}

float harrisResponse(int x, int y, const Mat& gx, const Mat& gy, short windowSize){
    Mat covarianceMatrix = Mat::zeros(2,2,CV_32FC1);
    float len = floor(windowSize/2),det, trace;
    float k = 0.04;
    for(int i=-len; i<=len; i++){
        for(int j=-len; j<=len; j++){
            covarianceMatrix.at<float>(0,0) += pow(gx.at<uchar>(x+i,y+j),2);
            covarianceMatrix.at<float>(0,1) += gx.at<uchar>(x+i,y+j)*gy.at<uchar>(x+i,y+j);
            covarianceMatrix.at<float>(1,0) += gx.at<uchar>(x+i,y+j)*gy.at<uchar>(x+i,y+j);
            covarianceMatrix.at<float>(1,1) += pow(gy.at<uchar>(x+i,y+j),2);
        }
    }
    det = covarianceMatrix.at<float>(0,0)*covarianceMatrix.at<float>(1,1)-covarianceMatrix.at<float>(0,1)*covarianceMatrix.at<float>(1,0);
    trace = covarianceMatrix.at<float>(0,0)+covarianceMatrix.at<float>(1,1);
    return det-k*(pow(trace,2));
}

void suppressCorners(list<Pixel> responses, Mat &inoutMatrix){
	responses.sort();
	int neighborhood = 11;
	int pad = floor(neighborhood/2);
	float range = sqrt(pow(pad,2)+pow(pad,2));
	list<Pixel>::iterator it,it2;
	for(it=responses.begin();it!=responses.end();){
		Pixel currentCorner = *it;
		it2 = it;
		advance(it2,1);
		for(;it2!=responses.end();){
			Pixel neighbor = *it2;
			float pixelDistance = sqrt(pow(neighbor.getRow()-currentCorner.getRow(),2)+pow(neighbor.getCol()-currentCorner.getCol(),2));
			if(pixelDistance<=range)
				it2 = responses.erase(it2);
			else
				++it2;
		}
		++it;
	}
	for(it=responses.begin(); it!=responses.end(); it++)
		circle(inoutMatrix, Point(it->getCol(),it->getRow()), pad, Scalar(150), 1, 4);
}

void harris(const Mat& image, const Mat& gx, const Mat& gy){
    //TO IMPLEMENT: MAXIMA SUPPRESSION
    int N = gx.rows;
    int M = gy.cols;  
	list<Pixel> responses;
	int neighborhood = 3;
	Mat beforeMatrix = image.clone();
    for(int row=PAD; row<N-PAD; row++){
        for(int col=PAD; col<M-PAD; col++){
			Pixel pixel(row,col,harrisResponse(row,col,gx,gy,neighborhood));
            if(pixel.getResponse()>10000){
               circle(beforeMatrix,Point(col,row),(neighborhood*2)-1, Scalar(150), 1, 4);
			   responses.push_front(pixel);
			}
		}
    }

    imshow("harrisBefore", beforeMatrix);
	
	Mat afterMatrix = image.clone();
	suppressCorners(responses, afterMatrix);
	
    imshow("harrisAfter", afterMatrix);

}

void displayHistogram(const Mat& image){
    int N = image.rows;
    int M = image.cols;
    vector<int> histogram(256,0);
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            histogram.at(image.at<uchar>(i,j))++;
    //FIND THE MAX VALUE OF THE BINS
    int max = histogram.at(0);
    for(int i = 1; i < 256; i++)
        if(max < histogram.at(i))
            max = histogram.at(i);
    int hist_width = 1024;
    int hist_height = 400;
    int bin_widht = round((float)hist_width/256);
    Mat histogramImage(hist_height+24, hist_width+48, CV_8UC1, Scalar(255, 255, 255));
    for(int i=0; i<256; i++)
        histogram.at(i) = ((double)histogram.at(i)/max)*hist_height;
    //DRAW THE LINE
    for(int i = 0; i < 256; i++)
      //line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
        line(histogramImage, Point(bin_widht*(i)+24, hist_height), Point(bin_widht*(i)+24, hist_height - histogram.at(i)),Scalar(0,0,0), 2, 8, 0);
    imshow("histogram", histogramImage);
    
}


int main(int argc, char** argv ){
	const char* fileName = (argc>=2)? argv[1]:"./scacchi.jpg";
	Mat sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);
    
	if (sourceMatrix.empty()){
        cout<<"--- EMPTY IMAGE ---"<<endl;
        exit(EXIT_FAILURE);
    }                         
    //Mat outputImage(sourceMatrix.size(),sourceMatrix.type());
    Mat paddedMatrix, histogram;
    
    createPaddedMatrix(sourceMatrix,paddedMatrix);
    Mat gx(paddedMatrix.size(), CV_8UC1);
    Mat gy(paddedMatrix.size(), CV_8UC1);
    sobel(paddedMatrix,gx,gy);
    gaussianFilter(gx);
    gaussianFilter(gy);
    //displayHistogram(outputImage);
    harris(paddedMatrix,gx,gy);
    waitKey(0);
    return 0;
}
*/