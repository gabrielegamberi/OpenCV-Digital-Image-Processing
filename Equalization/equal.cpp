#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <time.h>       /* time */
using namespace std;
using namespace cv;

void equalizeImage(Mat& sourceImage, Mat& outputImage){
    CV_Assert(sourceImage.depth() == CV_8U);  // accetto solo immagini uchar
    int L = 256;
    int M = sourceImage.rows;
    int N = sourceImage.cols;
    
    vector<int> pixelCount(L,0);
    for(int row=0; row<M; row++){ //Per ogni pixel conta le occorrenze di quel valore di pixel
        for(int col=0; col<N; col++){
            int pixelValue = saturate_cast<uchar>(sourceImage.at<uchar>(row,col));
            pixelCount.at(pixelValue)++;
        }
    }
    vector<int> newPixelValue(L,0);
    /*  calcola i nuovi valori di pixel: 
                Sk = (L-1) sommatoria(prob)                         da j a k  -> SK=(L-1)SUM(occorrenzePixel/MN)
        oppure: Sk = (L-1)/(M*N) * sommatoria(occorrenzePixel)      da j a k 
    */
    for(int k=0; k<pixelCount.size(); k++){
        double actualPixelValue = 0;
        for(int j=0; j<k; j++)
            actualPixelValue+=pixelCount.at(j);
        newPixelValue.at(k) = round(((double)(L-1)/(M*N))*actualPixelValue);
    }
    for(int row=0; row<M; row++)
        for(int col=0; col<N; col++)
            outputImage.at<uchar>(row,col) = newPixelValue.at(sourceImage.at<uchar>(row,col));
}

int main(int argc, char** argv)
{
    const char* fileName = (argc>=2)? argv[1]:"../waves.jpg";
    Mat sourceImage = imread(fileName, IMREAD_GRAYSCALE);
    if(sourceImage.empty()){
        cerr<<"--- NO DATA FOUND ---"<<endl;
        exit(EXIT_FAILURE);
    }
    Mat outputImage = sourceImage.clone();
    equalizeImage(sourceImage, outputImage);

    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", sourceImage);        
    namedWindow("output", WINDOW_AUTOSIZE);
    imshow("output", outputImage);
    waitKey(0);

    return 0;
}