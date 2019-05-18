//SELEZIONE MANUALE DEL RANGE TRAMITE RIGA DI COMANDO 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <limits.h>

//#define RADCOEF CV_PI/180
#define RADCOEF CV_PI/180

using namespace std;
using namespace cv;

int lowThreshold;

void HoughTransformCircle(Mat edgeMatrix, Mat& outputMatrix, int rmin, int rmax) {
	double theta;
	
	int radius = rmin;
	int step = 1;
	do{
		Mat vote = Mat::zeros(edgeMatrix.rows,edgeMatrix.cols, CV_8UC1);	//initialize to 0
		int xc, yc;
		for(int y=0; y<edgeMatrix.rows; y++){
			for(int x=0; x<edgeMatrix.cols; x++){
				if(edgeMatrix.at<uchar>(y,x) > 250){	
					for(int t=0; t<360; t++){
						theta = t*RADCOEF;
						xc = x-cvRound(radius*cos(theta));
						yc = y-cvRound(radius*sin(theta));
						if(xc<0 || xc>edgeMatrix.cols || yc<0 || yc>edgeMatrix.rows)
							continue;
						vote.at<uchar>(yc,xc)++;
					}
				}		
			}
		}
		string voteName = "vote";
		namedWindow(voteName+to_string(radius), WINDOW_AUTOSIZE);
		imshow(voteName+to_string(radius), vote);
		waitKey(0);

		for(yc=0; yc<vote.rows; yc++){
			for(xc=0; xc<vote.cols; xc++){
				if(vote.at<uchar>(yc,xc)>lowThreshold){	
					Point center = Point(xc,yc);
					circle(outputMatrix, center, 5, Scalar(0,255,0),2, 8,0);	
					circle(outputMatrix, center, radius, Scalar(0,0,255),2, 8,0);
				}
			}
		}
		radius+=step;
	}while(radius<=rmax);
}


int main(int argc, char** argv) {
	if(argc!=5){
		cerr<< "Usage: ./<programname> <imagename.format> <minradius> <maxradius> <minthreshold>" << endl;
		exit(EXIT_FAILURE);
	}
	Mat sourceMatrix = imread(argv[1], IMREAD_GRAYSCALE);
	int minRadius, maxRadius;
	minRadius = atoi(argv[2]);
	maxRadius = atoi(argv[3]);
	lowThreshold = atoi(argv[4]);
	
	if(sourceMatrix.empty()){
		cerr<< "Image format not valid." << endl;
		exit(EXIT_FAILURE);
	}
	
	Mat blurredMatrix, edgeMatrix;
	Mat outputMatrix = sourceMatrix.clone();
	
	cvtColor(outputMatrix, outputMatrix, COLOR_GRAY2BGR); //useful to color the image with red lines
	
	//Before calling Hough we have to apply some blurring to the image (dim: 5x5) to then use Canny Edge Detector
	GaussianBlur(sourceMatrix, blurredMatrix, Size(5,5), 1.44, 2);
	imshow("Gaussian Blur", blurredMatrix);

	Canny(blurredMatrix, edgeMatrix, 60, 150, 3); //(InMatrix, OutMatrix, Threshold #1 & 2 for hysteresis, dim. SobelKernel)
	imshow("Canny Edge", edgeMatrix);
	
	//Applying Hough transform
	HoughTransformCircle(edgeMatrix, outputMatrix, minRadius, maxRadius);
	
	//Showing the result
	imshow("Hough Circles", outputMatrix);
	waitKey(0);
	return(0);
}

/*
// Hough con parametri prefessati

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <limits.h>

//#define RADCOEF CV_PI/180
#define RADCOEF CV_PI/180

using namespace std;
using namespace cv;

int lowThreshold;

void HoughTransformCircle(Mat edgeMatrix, Mat& outputMatrix, int rmin, int rmax, int threshold) {
	cout<<"RMIN: "<<rmin<<"\tRMAX: "<<rmax<<endl;
	double theta;
	int radius = rmin;
	int step = 1;
	do{
		Mat vote = Mat::zeros(edgeMatrix.rows,edgeMatrix.cols, CV_8UC1);	//initialize to 0
		int xc, yc;
		for(int y=0; y<edgeMatrix.rows; y++){
			for(int x=0; x<edgeMatrix.cols; x++){
				if(edgeMatrix.at<uchar>(y,x) > 250){	
					for(int t=0; t<360; t++){
						theta = t*RADCOEF;
						xc = x-cvRound(radius*cos(theta));
						yc = y-cvRound(radius*sin(theta));
						if(xc<0 || xc>edgeMatrix.cols || yc<0 || yc>edgeMatrix.rows)
							continue;
						vote.at<uchar>(yc,xc)++;
					}
				}		
			}
		}
		string voteName = "vote";
		namedWindow(voteName+to_string(radius), WINDOW_AUTOSIZE);
		imshow(voteName+to_string(radius), vote);
		waitKey(0);

		for(yc=0; yc<vote.rows; yc++){
			for(xc=0; xc<vote.cols; xc++){
				if(vote.at<uchar>(yc,xc)>threshold){	
					Point center = Point(xc,yc);
					circle(outputMatrix, center, 5, Scalar(0,255,0),2, 8,0);	
					circle(outputMatrix, center, radius, Scalar(0,0,255),2, 8,0);
				}
			}
		}
		radius+=step;
	}while(radius<=rmax);
}


int main(int argc, char** argv) {

	if(argc!=5){
		cerr<< "Usage: ./<programname> <imagename.format> <minradius> <maxradius> <minthreshold>" << endl;
		exit(EXIT_FAILURE);
	}
	Mat sourceMatrix = imread(argv[1], IMREAD_GRAYSCALE);
	int minRadius, maxRadius;
	minRadius = atoi(argv[2]);
	maxRadius = atoi(argv[3]);
	lowThreshold = atoi(argv[4]);
	
	if(sourceMatrix.empty()){
		cerr<< "Image format not valid." << endl;
		exit(EXIT_FAILURE);
	}
	
	Mat blurredMatrix, edgeMatrix;
	Mat outputMatrix = sourceMatrix.clone();
	
	//Before calling Hough we have to apply some blurring to the image (dim: 5x5) to then use Canny Edge Detector
	GaussianBlur(sourceMatrix, blurredMatrix, Size(5,5), 1.44, 2);
	imshow("Gaussian Blur", blurredMatrix);

	Canny(blurredMatrix, edgeMatrix, 60, 150, 3); //(InMatrix, OutMatrix, Threshold #1 & 2 for hysteresis, dim. SobelKernel)
	imshow("Canny Edge", edgeMatrix);
	
	cvtColor(outputMatrix, outputMatrix, COLOR_GRAY2BGR); //useful to color the image with red lines
	
	//Applying Hough transform
	HoughTransformCircle(edgeMatrix, outputMatrix, 36, 38, 212);
	HoughTransformCircle(edgeMatrix, outputMatrix, 46, 48, 170);
	
	//Showing the result
	imshow("Hough Circles", outputMatrix);
	waitKey(0);
	return(0);
}
*/