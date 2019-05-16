#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace cv;
using namespace std;
#define SIZE 3
#define PAD floor(SIZE/2)

Mat horizPrewit = (Mat_<char>(SIZE,SIZE)<<-1,-1,-1,
                                            0, 0, 0,
                                            1, 1, 1
);
Mat vertPrewit =  (Mat_<char>(SIZE,SIZE)<<-1, 0, 1,
                                           -1, 0, 1,
                                           -1, 0, 1
);


void equalize(Mat& inputImage){
    int L = 256;
    int N = inputImage.rows;
    int M = inputImage.cols;

    vector<int> pixelCount(L,0);
    for(int row=0; row<N; row++) //count actual pixels
        for(int col=0; col<M; col++)
            pixelCount.at(inputImage.at<uchar>(row,col))++;

    vector<int> equalizedPixel(L,0);
    for(int k=0; k<L; k++){ //calculate new pixels values (formula)
        int sumPixel = 0;
        for(int j=0; j<k; j++)
          sumPixel+=pixelCount.at(j);
        equalizedPixel.at(k) = ((float)(L-1)/(M*N))*sumPixel;
    }

    for(int row=0; row<N; row++)  //insert new pixel into the image
        for(int col=0; col<M; col++)
            inputImage.at<uchar>(row,col) = equalizedPixel.at(inputImage.at<uchar>(row,col));

}

void prewitEdges(Mat& inputImage, Mat& outputImage){
    int N = inputImage.rows;
    int M = inputImage.cols;
    int threshold = 80;
    Mat vertEdges(inputImage.size(), inputImage.type());
    Mat horEdges(inputImage.size(), inputImage.type());

    equalize(inputImage);

    for(int row=PAD; row<N-PAD; row++){
        for(int col=PAD; col<M-PAD; col++){
            int horAccum, vertAccum;
            horAccum = vertAccum = 0;
            for(int i=-PAD; i<=PAD; i++){
                for(int j=-PAD; j<=PAD; j++){
                  horAccum+=(int)(inputImage.at<uchar>(row+i,col+j)*horizPrewit.at<char>(i+PAD, j+PAD));
                  vertAccum+=(int)(inputImage.at<uchar>(row+i,col+j)*vertPrewit.at<char>(i+PAD, j+PAD));
              }
            }
            horAccum = (horAccum>threshold)? 255:0;
            vertAccum = (vertAccum>threshold)? 255:0;
            outputImage.at<uchar>(row,col) = max(horAccum,vertAccum);
        }
    }
    namedWindow("outputImage", WINDOW_AUTOSIZE);
    imshow("outputImage", outputImage);
}


int main(int argc, char** argv )
{
    const char* fileName = (argc>=2)? argv[1]:"../lena.jpg";
    cout<<"ARGC: "<<argc<<endl;
    cout<<"FILE: "<<fileName<<endl;
    Mat rawImage = imread( fileName, IMREAD_GRAYSCALE);
    if(rawImage.empty()){
        cerr<<"--- NO DATA FOUND ---"<<endl;
        exit(EXIT_FAILURE);
    }

    Mat outputImage(rawImage.size(), rawImage.type());


    prewitEdges(rawImage, outputImage);

    namedWindow("input", WINDOW_AUTOSIZE);
    imshow("input", rawImage);
    waitKey(0);
    return 0;
}
