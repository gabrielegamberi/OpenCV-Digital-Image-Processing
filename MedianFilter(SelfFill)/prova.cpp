#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

/* RICORDA
Mat kernel(LEN,LEN,CV_8U,Scalar(1));
Mat kernel = Mat::ones(LEN,LEN,CV_8U);
Mat mask = (Mat_<char>(LEN,LEN)<<1,1,1
                                 1,1,1
                                 1,1,1
);
*/

#define SIZE 5
#define PAD floor(SIZE/2)

int getMedian(int value[]){
    int n = SIZE*SIZE;
    int iMin;
    for(int i=0; i<n-1; i++){
        iMin = i;
        for(int j=i+1; j<n; j++)
            if(value[j]<value[iMin])
                iMin = j;
		int temp = value[iMin];
		value[iMin] = value[i];
		value[i] = temp;
	}
    return value[n/2];
}

void medianFilter(const Mat paddedMatrix, Mat& outputMatrix){
	int maskedValue[SIZE*SIZE];
    for(int y=+PAD; y<paddedMatrix.rows-PAD; y++){
        for(int x=+PAD; x<paddedMatrix.cols-PAD; x++){
			int mask_iter = 0;
			for(int row=-PAD; row<=PAD; row++)
				for(int col=-PAD; col<=PAD; col++)
					maskedValue[mask_iter++] = paddedMatrix.at<uchar>(row+y, col+x);
            outputMatrix.at<uchar>(y-PAD,x-PAD) = getMedian(maskedValue);    //assegno al pixel corrente la media dei pixel nell'intorno
        }
    }
}

void buildPaddedMatrix(const Mat inputMatrix, Mat& paddedMatrix){
	int corner = 0;
	int row, col;
	paddedMatrix = Mat::zeros(inputMatrix.rows+2*PAD, inputMatrix.cols+2*PAD, inputMatrix.type());
	for(row=PAD; row<paddedMatrix.rows-PAD; row++){
		for(col=PAD; col<paddedMatrix.cols-PAD; col++){
			paddedMatrix.at<uchar>(row,col) = inputMatrix.at<uchar>(row-PAD, col-PAD);
		}
	}
	while(corner<4){
		switch(corner){
			case 0://copia colonna di sinistra
				for(row=PAD; row<paddedMatrix.rows-PAD; row++)
					for(col=0; col<PAD; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(row,PAD);
				break;
			case 1://colonna di destra
				for(row=PAD; row<paddedMatrix.rows-PAD; row++)
					for(col=paddedMatrix.cols-PAD; col<paddedMatrix.cols; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(row,paddedMatrix.cols-PAD-1);
				break;
			case 2://riga superiore
				for(row=0; row<PAD; row++)
					for(col=0; col<paddedMatrix.cols; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(PAD,col);
				break;
			case 3://riga inferiore
				for(row=paddedMatrix.rows-PAD; row<paddedMatrix.rows; row++)
					for(col=0; col<paddedMatrix.cols; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(paddedMatrix.rows-PAD-1, col);
				break;
		}

		corner++;
	}


}

int main(int argc, char** argv )
{
   	const char* fileName = (argc>=2)? argv[1]:"./lena.jpg";
	Mat inputMatrix = imread(fileName, IMREAD_GRAYSCALE);
	if(inputMatrix.empty()){
		cerr<<"-- Matrix is empty --"<<endl;
		exit(EXIT_FAILURE);
	}

	Mat paddedMatrix, noPaddedMatrix, outputMatrix;
	buildPaddedMatrix(inputMatrix, paddedMatrix);

	outputMatrix = Mat(inputMatrix.size(), inputMatrix.type());

	medianFilter(paddedMatrix, outputMatrix);
	
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", inputMatrix);
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", outputMatrix);
	waitKey(0);
	return 0;
}
