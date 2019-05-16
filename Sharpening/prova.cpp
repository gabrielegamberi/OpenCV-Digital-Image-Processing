#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


    Mat kernel = (Mat_<char>(3,3) <<  0, -1,  0,
                                     -1,  5, -1,
                                      0, -1,  0);

void sharpen(const Mat& myImage, Mat& resultImage){
    CV_Assert(myImage.depth()==CV_8U);      //accetto solo immagini uchar
    const int nChannels = myImage.channels();
    resultImage.create(myImage.size(),myImage.type());

    for(unsigned int i=1; i<myImage.rows-1; i++){
        const uchar* previous = myImage.ptr<uchar>(i-1);
        const uchar* current  = myImage.ptr<uchar>(i  );
        const uchar* next     = myImage.ptr<uchar>(i+1);
        uchar* output = resultImage.ptr<uchar>(i);      //posizionati alla riga i-sima
        for(unsigned int j=nChannels; j<nChannels*(myImage.cols-1); j++) //con j=nChannels mi posiziono sul canale posizionato sulla 2° colonna
            *(output++) = saturate_cast<uchar>(current[j]           *   kernel.at<char>(1,1)
                                              +current[j-nChannels] *   kernel.at<char>(1,0)
                                              +current[j+nChannels] *   kernel.at<char>(1,2)
                                              +previous[j]          *   kernel.at<char>(0,1)
                                              +next[j]              *   kernel.at<char>(2,1)
                                              );
    }

    resultImage.row(0).setTo(Scalar(0));
    resultImage.row(resultImage.rows-1).setTo(Scalar(0));
    resultImage.col(0).setTo(Scalar(0));
    resultImage.col(resultImage.cols-1).setTo(Scalar(0));
}

int main(int argc, char** argv )
{
    const char* filename = argc>=2? argv[1]:"../lena.jpg";
    const char* inputWindow = "input", *outputWindow = "output";
    Mat source, destination;
    source = imread(filename, IMREAD_COLOR);
    if(source.empty()){
        cout<<"--- ERRORE --- no data found"<<endl;
        exit(EXIT_FAILURE);
    }

    namedWindow(inputWindow, WINDOW_AUTOSIZE);
    imshow(inputWindow, source);
    sharpen(source, destination);
    
    namedWindow(outputWindow, WINDOW_AUTOSIZE);
    imshow(outputWindow, destination);
    
    waitKey(0);
    return 0;
}