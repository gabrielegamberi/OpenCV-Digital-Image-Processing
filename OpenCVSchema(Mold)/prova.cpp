#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main(int argc, char** argv )
{
    const char* filename = argc>=2? argv[1]:"../lena.jpg";
    Mat matrix = imread(filename,1);

    if (matrix.empty()){
        cout<<"-- ERROR -- no data"<<endl;
        exit(EXIT_FAILURE);
    }
    namedWindow("Display Image", WINDOW_AUTOSIZE );
    imshow("Display Image", matrix);
    waitKey(0);

    return 0;
}