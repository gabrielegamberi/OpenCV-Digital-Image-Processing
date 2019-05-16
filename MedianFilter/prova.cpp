#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define LEN 5
#define PAD floor(LEN/2)

//Ordino l'array tramite il Selection Sort e restituisco il valore centrale dell'array ordinato (mediano)
int getMedian(int value[]){
    int n = LEN*LEN;
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

void medianFilter(Mat& inputMatrix, Mat& outputMatrix){
    outputMatrix = Mat(inputMatrix.size(), inputMatrix.type());
    Mat paddedMatrix = Mat::zeros(inputMatrix.rows+2*PAD, inputMatrix.cols+2*PAD, inputMatrix.type());
    for(int row=PAD; row<paddedMatrix.rows-PAD; row++)  //inizializzo la matrice padded ricopiando quella di input
        for(int col=PAD; col<paddedMatrix.cols-PAD; col++)
            paddedMatrix.at<uchar>(row,col) = inputMatrix.at<uchar>(row-PAD,col-PAD);
    int maskedValue[LEN*LEN];
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

int main(int argc, char** argv )
{
    const char* fileName = argc>=2? argv[1]:"../lena.jpg";
    const char* inputWindowName = "Input Picture";

    Mat oMatrix = imread(fileName, IMREAD_GRAYSCALE);
    Mat medianMatrix;

    if(oMatrix.empty()){
        cout<<"-- ERROR -- Matrix is empty";
        exit(EXIT_FAILURE);
    }

    namedWindow(inputWindowName, WINDOW_AUTOSIZE);
    imshow(inputWindowName, oMatrix);
    waitKey(0);

    medianFilter(oMatrix, medianMatrix);
	namedWindow("Median Matrix", WINDOW_AUTOSIZE);
    imshow("Median Matrix", medianMatrix);
    waitKey(0);
    return 0;
}