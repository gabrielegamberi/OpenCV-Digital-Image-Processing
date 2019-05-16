#include <stdio.h>
#include <iostream>
#include <string>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define LEN 3
#define PAD floor(LEN/2)

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
        filter = getMin;
        initValue = UCHAR_MAX;
    }else{
        filter = getMax;
        initValue = 0;
    }
    for(int row=PAD; row<tempMatrix.rows-PAD; row++){
        for(int col=PAD; col<tempMatrix.cols-PAD; col++){
            int replacingValue = initValue;
            for(int i=-PAD; i<=PAD; i++){
                for(int j=-PAD; j<=PAD; j++){
                    int currentPixelVal = tempMatrix.at<uchar>(row+i,col+i);
                    replacingValue = filter(currentPixelVal, replacingValue);
                    //cout<<"CurrentPixel: "<<currentPixelVal<<endl;
                }
            }
            //cout<<"Replacing Val: "<<replacingValue<<endl;
            //exit(1);
            inoutPaddedMatrix.at<uchar>(row,col) = replacingValue;
        }
    }
}

void createPaddedMatrix(const Mat inputMatrix, Mat& paddedMatrix){
    paddedMatrix = Mat::zeros(inputMatrix.rows+2*PAD, inputMatrix.cols+2*PAD, inputMatrix.type());
    for(int i=PAD; i<paddedMatrix.rows-PAD; i++)
        for(int j=PAD; j<paddedMatrix.cols-PAD; j++)
            paddedMatrix.at<uchar>(i,j) = inputMatrix.at<uchar>(i-PAD, j-PAD);
    
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
    const char* fileName = argc>=2? argv[1]:"../lena.jpg";
    const char* inputWindowName = "Input Picture";
    string selectedMode;
    Mat inputMatrix = imread(fileName, IMREAD_GRAYSCALE);
    Mat noPaddedMatrix, paddedMatrix;
    
    if(inputMatrix.empty()){
        cout<<"-- ERROR -- Matrix is empty";
        exit(EXIT_FAILURE);
    }

    //cout<<"Do you want to erode or dilate? ";
    //getline(cin, selectedMode);

    createPaddedMatrix(inputMatrix, paddedMatrix); //ricopia i pixel nel padding
    
    namedWindow(inputWindowName, WINDOW_AUTOSIZE);
    imshow(inputWindowName, inputMatrix);
    
    erode_dilate_matrix(paddedMatrix, "erode");
    erode_dilate_matrix(paddedMatrix, "dilate");
    
    paddedMatrix(Rect(Point(PAD, paddedMatrix.cols-PAD), Point(paddedMatrix.rows-PAD, PAD))).copyTo(noPaddedMatrix);
    
    namedWindow("Output Matrix", WINDOW_AUTOSIZE);
    imshow("Output Matrix", noPaddedMatrix);
    waitKey(0);
    return 0;
}