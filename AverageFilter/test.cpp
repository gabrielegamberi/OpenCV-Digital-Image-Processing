#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

#define LEN 5
#define PAD floor(LEN/2)

Mat kernel(LEN,LEN,CV_8U,Scalar(1));
//Mat kernel = Mat::ones(LEN,LEN,CV_8U);
/*Oppure
Mat mask = (Mat_<char>(LEN,LEN)<<1,1,1
                                 1,1,1
                                 1,1,1
);*/

void getMaskedValues(int currentY, int currentX, Mat inputMatrix, int* maskedValue){
    int i=0;
    for(int row=-PAD; row<=PAD; row++)
        for(int col=-PAD; col<=PAD; col++)
            maskedValue[i++] = inputMatrix.at<uchar>(row+currentY, col+currentX)*kernel.at<uchar>(row+PAD,col+PAD);
}
void meanFilter(Mat& inputMatrix, Mat& meanMatrix){
    int pad = PAD;
    meanMatrix = inputMatrix.clone();
    Mat paddedMatrix = Mat::zeros(inputMatrix.rows+2*pad, inputMatrix.cols+2*pad, inputMatrix.type());
    for(int row=pad; row<paddedMatrix.rows-pad; row++)  //inizializzo la matrice padded ricopiando quella di input
        for(int col=pad; col<paddedMatrix.cols-pad; col++)
            paddedMatrix.at<uchar>(row,col) = inputMatrix.at<uchar>(row-pad,col-pad);
    int denominator = 0;
    for(int i=0; i<kernel.rows; i++)
        for(int j=0; j<kernel.cols; j++)
            denominator+=kernel.at<uchar>(i,j);
    int maskedValue[LEN*LEN];
    for(int y=+pad; y<paddedMatrix.rows-pad; y++){
        for(int x=+pad; x<paddedMatrix.cols-pad; x++){
            getMaskedValues(y,x,paddedMatrix,maskedValue);
            int avMiddleValue = 0;
            for(int i=0; i<LEN*LEN; i++)
                avMiddleValue+=maskedValue[i];
            avMiddleValue/=denominator;
            meanMatrix.at<uchar>(y-pad,x-pad) = avMiddleValue;    //assegno al pixel corrente la media dei pixel nell'intorno
        }
    }
    namedWindow("Mean Matrix", WINDOW_AUTOSIZE);
    imshow("Mean Matrix", meanMatrix);
    waitKey(0);
}

int main(int argc, char** argv )
{
    const char* fileName = argc>=2? argv[1]:"../lena.jpg";
    const char* inputWindowName = "Input Picture";

    Mat oMatrix = imread(fileName, IMREAD_GRAYSCALE);
    Mat meanMatrix;

    if(oMatrix.empty()){
        cout<<"-- ERROR -- Matrix is empty";
        exit(EXIT_FAILURE);
    }

    namedWindow(inputWindowName, WINDOW_AUTOSIZE);
    imshow(inputWindowName, oMatrix);
    waitKey(0);

    meanFilter(oMatrix, meanMatrix);

    return 0;
}