#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define OVERALL_MATRIX_PAD 2
#define SOBEL_SIZE 3
#define SOBEL_PAD 1

Mat horSobel = (Mat_<char>(SOBEL_SIZE, SOBEL_SIZE)<<-1,-2,-1,
												     0, 0, 0,
													 1, 2, 1
);
Mat vertSobel = (Mat_<char>(SOBEL_SIZE, SOBEL_SIZE)<<-1, 0, 1,
												     -2, 0, 2,
													 -1, 0, 1
);

int N, M;

class Pixel{
	public:
		int row;
		int col;
		int lowestEigenVal;

		Pixel(int y, int x, int eigen):row(y),col(x),lowestEigenVal(eigen){}

		bool operator<(const Pixel &nextPixel)const{
			return (nextPixel.lowestEigenVal<lowestEigenVal);
		}
};


void calcSobelMatrixes(const Mat& inputMatrix, Mat& sobelX, Mat& sobelY){
	sobelX = Mat::zeros(N,M,CV_8UC1);
	sobelY = Mat::zeros(N,M,CV_8UC1);

	int pad = OVERALL_MATRIX_PAD;

	int xGrad, yGrad;
	for(int row=pad; row<N-pad; row++){
		for(int col=pad; col<M-pad; col++){
			xGrad = yGrad = 0;
			for(int i=-SOBEL_PAD; i<=SOBEL_PAD; i++){
				for(int j=-SOBEL_PAD; j<=SOBEL_PAD; j++){
					xGrad+=inputMatrix.at<uchar>(row+i,col+j)*vertSobel.at<char>(i+SOBEL_PAD, j+SOBEL_PAD);
					yGrad+=inputMatrix.at<uchar>(row+i,col+j)*horSobel.at<char>(i+SOBEL_PAD, j+SOBEL_PAD);
				}
			}
			sobelX.at<uchar>(row,col) = saturate_cast<uchar>(abs(xGrad));
			sobelY.at<uchar>(row,col) = saturate_cast<uchar>(abs(yGrad));
		}
	}
}

float getEigenValue(int row, int col, const Mat& sobelX, const Mat& sobelY, int neighborHood){
	int windowPad = floor(neighborHood/2);
	Mat covarianceMatrix = Mat::zeros(2,2,CV_32FC1);
	for(int i=-windowPad; i<=windowPad; i++){
		for(int j=-windowPad; j<=windowPad; j++){
			covarianceMatrix.at<float>(0,0)+=(float)pow(sobelX.at<uchar>(row+i,col+j),2);
			covarianceMatrix.at<float>(0,1)+=(float)sobelX.at<uchar>(row+i,col+j)*sobelY.at<uchar>(row+i,col+j);
			covarianceMatrix.at<float>(1,0)+=(float)sobelX.at<uchar>(row+i,col+j)*sobelY.at<uchar>(row+i,col+j);
			covarianceMatrix.at<float>(1,1)+=(float)pow(sobelY.at<uchar>(row+i,col+j),2);
		}
	}
	Mat eVect;
	eigen(covarianceMatrix, eVect);
	float l1, l2, minLambda;
	l1 = eVect.at<float>(0,0);
	l2 = eVect.at<float>(1,0);
	minLambda = (l1<l2)? l1:l2;
	if(minLambda>1000)
		return minLambda;
	else
		return 0; 
}

void suppressCorners(list<Pixel> corners, Mat& resultMatrix){
	list<Pixel>::iterator it, it2;
	int neighborhood = 15;
	int pad = floor(neighborhood/2);
	float limit = sqrt(pow(pad,2)+pow(pad,2));
	corners.sort();
	for(it = corners.begin(); it!=corners.end();){
		Pixel currentCorner = *it;
		it2 = it;
		advance(it2, 1);
		for(;it2!=corners.end();){
			Pixel neighborCorner = *it2;
			float distance = sqrt(pow(neighborCorner.row-currentCorner.row,2)+pow(neighborCorner.col-currentCorner.col,2));
			if(distance<=limit){
				it2 = corners.erase(it2);
			}else{
				++it2;
			}
		}
		++it;
	}

	for(it = corners.begin(); it!=corners.end(); it++)
		circle(resultMatrix, Point((*it).col, (*it).row), pad, Scalar(150), 1, LINE_AA);
	
}

void detectHarrisCorners(const Mat &grayImage){
	Mat paddedMatrix = Mat::zeros(grayImage.rows+2*OVERALL_MATRIX_PAD, grayImage.cols+2*OVERALL_MATRIX_PAD, grayImage.type());

	N = paddedMatrix.rows;
	M = paddedMatrix.cols;
	int pad = OVERALL_MATRIX_PAD;
	for(int row=pad; row<N-pad; row++)
		for(int col=pad; col<M-pad; col++)
			paddedMatrix.at<uchar>(row,col) = grayImage.at<uchar>(row-pad, col-pad);
	
	Mat sobelX, sobelY;

	calcSobelMatrixes(paddedMatrix, sobelX, sobelY);

	GaussianBlur(sobelX, sobelX, Size(3,3), 3);
	GaussianBlur(sobelY, sobelY, Size(3,3), 3);

	list<Pixel> corners;
	int neighborHood = min(pad, 5);
	Mat tempMatrix = paddedMatrix.clone();
	for(int row=pad; row<N-pad; row++){
		for(int col=pad; col<M-pad; col++){
			Pixel corner(row,col,getEigenValue(row, col, sobelX, sobelY, neighborHood));
			if(corner.lowestEigenVal!=0){
				corners.push_back(corner);
			}
		}
	}

	suppressCorners(corners, paddedMatrix);
	
	Mat resultMatrix = paddedMatrix.clone();
	//ricopia la matrice senza il padding | 	Point(x,y)!!
	paddedMatrix(Rect(Point(paddedMatrix.cols-1,OVERALL_MATRIX_PAD), Point(OVERALL_MATRIX_PAD,paddedMatrix.rows-1))).copyTo(resultMatrix);

	namedWindow("sobelX", WINDOW_AUTOSIZE);
	imshow("sobelX", sobelX);
	namedWindow("sobelY", WINDOW_AUTOSIZE);
	imshow("sobelY", sobelY);
	namedWindow("tempMatrix", WINDOW_AUTOSIZE);
	imshow("tempMatrix", tempMatrix);
	namedWindow("resultMatrix", WINDOW_AUTOSIZE);
	imshow("resultMatrix", resultMatrix);
	waitKey(0);
}



int main(int argc, char** argv ){
	if(argc!=2){
		cerr<<"--- ERROR --- current usage: <exe> <file.ext>"<<endl;
		exit(EXIT_FAILURE);
	}
   	const char* fileName = argv[1];
	Mat sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);
	if(sourceMatrix.empty()){
		cerr<<"-- Matrix is empty --"<<endl;
		exit(EXIT_FAILURE);
	}
	namedWindow("source", WINDOW_AUTOSIZE);
	imshow("source", sourceMatrix);
	detectHarrisCorners(sourceMatrix);
	return 0;
}
