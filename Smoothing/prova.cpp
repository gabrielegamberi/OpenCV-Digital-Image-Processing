//# ================ FILTRO SMOOTHING PASSA BASSO - MEDIANO, MEDIA e GAUSS (Media Pesata) ======================
// finestra 3 x 3
#include <stdlib.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

const int lenMask = 3;


void swap(int& a, int& b){
    int temp = a;
    a = b;
    b = temp;
}

//Ordino tramite il Selection Sort e restituisco il valore centrale (mediano)
int getMedian(int value[]){
    int n = lenMask*lenMask;
    int iMin;
    for(int i=0; i<n-1; i++){
        iMin = i;
        for(int j=i+1; j<n; j++)
            if(value[j]<value[iMin])
                iMin = j;

        swap(value[iMin], value[i]);
    }
    return value[n/2];
}

void medianFilter(const Mat& src){
    Mat medianMatrix = src.clone();
    int value[lenMask*lenMask];
    for(unsigned int y=1; y<src.rows-1; y++){
        for(unsigned int x=1; x<src.cols-1; x++){
            value[0] = src.at<uchar>(y-1,   x-1);
            value[1] = src.at<uchar>(y-1,   x);
            value[2] = src.at<uchar>(y-1,   x+1);
            value[3] = src.at<uchar>(y,     x-1);
            value[4] = src.at<uchar>(y,     x);
            value[5] = src.at<uchar>(y,     x+1);
            value[6] = src.at<uchar>(y+1,   x-1);
            value[7] = src.at<uchar>(y+1,   x);
            value[8] = src.at<uchar>(y+1,   x+1);

            medianMatrix.at<uchar>(y,x) = getMedian(value);
        }
    }
    namedWindow("Median", CV_WINDOW_AUTOSIZE);
    imshow("Median", medianMatrix);
    waitKey(0);
}

void meanFilter(Mat src){
    Mat meanMatrix = src.clone();
    int n = lenMask*lenMask;
    Mat mask(3,3,CV_8U,Scalar(1));          //maschera costituita da soli 1
    int denominator = 0;
    for(unsigned int i=0; i<mask.rows; i++)
        for(unsigned int j=0; j<mask.cols; j++)
            denominator+=mask.at<char>(i,j);
    cout<<"DENOMINATOR: "<<denominator<<endl;
    int value[n];
    for(unsigned int y = 1; y < src.rows-1; y++)
        for(unsigned int x = 1; x < src.cols-1; x++){
            value[0] = src.at<uchar>(y-1, x-1)*mask.at<char>(0,0);
            value[1] = src.at<uchar>(y-1, x)*  mask.at<char>(0,1);
            value[2] = src.at<uchar>(y-1, x+1)*mask.at<char>(0,2);
            value[3] = src.at<uchar>(y, x-1)*  mask.at<char>(1,0);
            value[4] = src.at<uchar>(y, x)*    mask.at<char>(1,1);
            value[5] = src.at<uchar>(y, x+1)*  mask.at<char>(1,2);
            value[6] = src.at<uchar>(y+1, x-1)*mask.at<char>(2,0);
            value[7] = src.at<uchar>(y+1, x)*  mask.at<char>(2,1);
            value[8] = src.at<uchar>(y+1, x+1)*mask.at<char>(2,2);

            int tempAverage = 0;
            for(unsigned int i=0; i<n; i++)
                tempAverage+=value[i];
            tempAverage = tempAverage/denominator;
            meanMatrix.at<uchar>(y, x)=tempAverage;
        }

    namedWindow("Media", CV_WINDOW_AUTOSIZE);
    imshow("Media", meanMatrix);
    waitKey(0);
}


//MMMH NON SONO PER NIENTE CONVINTO
void gaussianFilter(Mat& src) {
    Mat gauss = src.clone();
    int n = lenMask*lenMask;
    Mat mask = (Mat_<char>(3,3) <<      1,  2,  1,
                                        2,  8,  2,
                                        1,  2,  1);
    int denominator = 0;
    for(unsigned int i=0; i<lenMask; i++)
        for(unsigned int j=0; j<lenMask; j++)
            denominator+=mask.at<char>(i,j);

    int value[lenMask * lenMask];
    for(unsigned int y = 1; y < src.rows-1; y++)
        for (unsigned int x = 1; x < src.cols-1; x++) {
            value[0] = src.at<uchar>(y-1, x-1) *    mask.at<char>(0,0);
            value[1] = src.at<uchar>(y-1, x)   *    mask.at<char>(0,1);
            value[2] = src.at<uchar>(y-1, x+1) *    mask.at<char>(0,2);
            value[3] = src.at<uchar>(y,   x-1) *    mask.at<char>(1,0);
            value[4] = src.at<uchar>(y,   x)   *    mask.at<char>(1,1);
            value[5] = src.at<uchar>(y,   x+1) *    mask.at<char>(1,2);
            value[6] = src.at<uchar>(y+1, x-1) *    mask.at<char>(2,0);
            value[7] = src.at<uchar>(y+1, x)   *    mask.at<char>(2,1);
            value[8] = src.at<uchar>(y+1, x+1) *    mask.at<char>(2,2);

            int tempAverage = 0;
            for(unsigned int i=0; i<n; i++)
                tempAverage+=value[i];
            tempAverage = tempAverage/denominator;
            gauss.at<uchar>(y, x)=tempAverage;
        }

    namedWindow("Gauss", CV_WINDOW_AUTOSIZE);
    imshow("Gauss", gauss);
    waitKey(0);
}


int main(int argc, char** argv) {
    const char* filename = argc>=2? argv[1]:"../lena.jpg";
    Mat src = imread(filename, IMREAD_GRAYSCALE);
    namedWindow("Src", CV_WINDOW_AUTOSIZE);
    imshow("Src", src);

    waitKey(0);
    //medianFilter(src);        //filtro mediano    (prendi il valore centrale)
    //meanFilter(src);          //filtro media      (fai la media dei pixel nell'intorno)
	//gaussianFilter(src);      //filtro gaussiano

    return 0;
}