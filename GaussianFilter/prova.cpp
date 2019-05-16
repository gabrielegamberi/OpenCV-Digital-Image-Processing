#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>

using namespace std;
using namespace cv;

#define SIZE 5
#define PAD floor(SIZE/2)


Mat kernel(SIZE,SIZE,CV_32FC1,Scalar(0));
/*Mat kernel = Mat::ones(LEN,LEN,CV_8U);
Mat mask = (Mat_<char>(LEN,LEN)<<1,1,1
                                 1,1,1
                                 1,1,1
);
*/

void buildGaussianFilter(){
	int variance = 3;
	int rowShift, colShift, B;
	float expTerm, minTerm;
	
	for(int i=PAD; i<kernel.rows; i++)
		for(int j=PAD; j<kernel.cols; j++)
			kernel.at<float>(i,j) = exp(-(pow(i-PAD,2)+pow(j-PAD,2))/(2*variance));
	minTerm = kernel.at<float>(kernel.rows-1,kernel.cols-1);
	
	for(int i=PAD; i<kernel.rows; i++){
		for(int j=PAD; j<kernel.cols; j++){
			B = round(kernel.at<float>(i,j)/minTerm);
			
			kernel.at<float>(i,j) = B;

			rowShift = PAD+(PAD-i); 
			colShift = PAD+(PAD-j);

			if(i==PAD){ //se mi trovo sulla riga centrale (scopia a specchio verso sinistra)
				kernel.at<float>(i,colShift) = B;
			}else if(j==PAD){ //se mi trovo sulla colonna centrale (copia a specchio verso l'alto)
				kernel.at<float>(rowShift,j) = B;
			}else{
				kernel.at<float>(rowShift,j) = B;
				kernel.at<float>(i,colShift) = B;
				kernel.at<float>(rowShift,colShift) = B;
			}
		}
	}

	for(int i=0; i<kernel.rows; i++){
		for(int j=0; j<kernel.cols; j++){
			int value = kernel.at<float>(i,j);
			cout<<value<<"\t";
		}
		cout<<endl;
	}
}

int main(int argc, char** argv ){
   	const char* fileName = (argc>=2)? argv[1]:"./lena.jpg";
	Mat inputMatrix = imread(fileName, IMREAD_GRAYSCALE);
	if(inputMatrix.empty()){
		cerr<<"-- Matrix is empty --"<<endl;
		exit(EXIT_FAILURE);
	}

	buildGaussianFilter();
	
	waitKey(0);
	return 0;
}
