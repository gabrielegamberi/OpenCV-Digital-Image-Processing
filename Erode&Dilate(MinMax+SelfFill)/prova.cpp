#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

#define SIZE 5
#define PAD floor(SIZE/2)

inline int getMax(int a, int b){
	return ((a>b)? (a):(b));
}
inline int getMin(int a, int b){
	return ((a<b)? (a):(b));
}

void erode_dilate_matrix(Mat& inoutPaddedMatrix, String selectedMode){
	Mat tempMatrix = inoutPaddedMatrix.clone();
	int initValue;
	int (*filter)(int,int);
	if(selectedMode == "erode"){
		initValue = UCHAR_MAX;
		filter = getMin;
	}else{
		initValue = 0;
		filter = getMax;
	}
	for(int row=PAD; row<tempMatrix.rows-PAD; row++){
		for(int col=PAD; col<tempMatrix.cols-PAD; col++){
			int replacingValue = initValue;
			for(int i=-PAD; i<=PAD; i++){
				for(int j=-PAD; j<=PAD; j++){
					int pixelVal = tempMatrix.at<uchar>(row+i,col+j);
					replacingValue = filter(replacingValue, pixelVal);
				}
			}
			inoutPaddedMatrix.at<uchar>(row,col) = replacingValue;
		}
	}
}


void selfFill(Mat& paddedMatrix){
	int corner = 0;
	int i,j;
	while(corner<4){
		switch(corner){
			case 0://copy left-column
				for(i=PAD; i<paddedMatrix.rows-PAD; i++)
					for(j=0; j<PAD; j++)
						paddedMatrix.at<uchar>(i,j) = paddedMatrix.at<uchar>(i,PAD);
				break;
			case 1://copy right-column
				for(i=PAD; i<paddedMatrix.rows-PAD; i++)
					for(j=paddedMatrix.cols-PAD; j<paddedMatrix.cols; j++)
						paddedMatrix.at<uchar>(i,j) = paddedMatrix.at<uchar>(i,paddedMatrix.cols-PAD-1);
				break;
			case 2://copy top-row (including cols)
				for(i=0; i<PAD; i++)
					for(j=0; j<paddedMatrix.cols; j++)
						paddedMatrix.at<uchar>(i,j) = paddedMatrix.at<uchar>(PAD,j);
				break;
			case 3://copy bottom-row (including cols)
				for(i=paddedMatrix.rows-PAD; i<paddedMatrix.rows; i++)
					for(j=0; j<paddedMatrix.cols; j++)
						paddedMatrix.at<uchar>(i,j) = paddedMatrix.at<uchar>(paddedMatrix.rows-PAD-1, j);
				break;
		}
		corner++;
	}
}

int main(int argc, char** argv )
{
    const char* filename = argc>=2? argv[1]:"../lena.jpg";
    Mat inputMatrix = imread(filename,IMREAD_GRAYSCALE);
	Mat paddedMatrix, noPaddedMatrix;
	string selectedMode;
	if (inputMatrix.empty()){
		cout<<"-- ERROR -- no data"<<endl;
		exit(EXIT_FAILURE);
	}
	paddedMatrix = Mat::zeros(inputMatrix.rows+2*PAD, inputMatrix.cols+2*PAD, inputMatrix.type());
	for(int i=PAD; i<paddedMatrix.rows-PAD; i++)
		for(int j=PAD; j<paddedMatrix.cols-PAD; j++)
			paddedMatrix.at<uchar>(i,j) = inputMatrix.at<uchar>(i-PAD, j-PAD);
	//selfFill(paddedMatrix);
	
	namedWindow("Display Image", WINDOW_AUTOSIZE );
	imshow("Display Image", inputMatrix);
	//cout<<"Do you want to erode or dilate? ";
	//getline(cin, selectedMode);
	erode_dilate_matrix(paddedMatrix, "erode");
	//erode_dilate_matrix(paddedMatrix, "dilate");
	paddedMatrix(Rect(Point(PAD, paddedMatrix.cols-PAD),Point(paddedMatrix.rows-PAD, PAD))).copyTo(noPaddedMatrix);

	namedWindow("Output", WINDOW_AUTOSIZE);
	imshow("Output", noPaddedMatrix);
	waitKey(0);
    return 0;
}
