#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <list>
#include <limits.h>

using namespace cv;
using namespace std;

class DistanceTransform{
	private:
		Mat matrix;
		Mat outputMatrix;
		int N;
		int M;
	
	void showNormalizedOutputMatrix(){
		Mat normalizedMatrix = Mat::zeros(outputMatrix.size(), outputMatrix.type());
		cv::normalize(outputMatrix, normalizedMatrix, 0, 255, NORM_MINMAX, CV_8UC1);
		namedWindow("normalized", WINDOW_AUTOSIZE);
		imshow("normalized", normalizedMatrix);
	}


	public:
		DistanceTransform(Mat inputMatrix):matrix(inputMatrix){
			N = inputMatrix.rows;
			M = inputMatrix.cols;
		}
		
		Mat& applyTransform(){
			outputMatrix = this->matrix.clone();
			//prima scansione (alto->basso & sinistra->destra)
			for(int row=1; row<N-1; row++){
				for(int col=1; col<M-1; col++){
					int actualPixelValue = outputMatrix.at<uchar>(row,col);
					if(actualPixelValue!=0){
						uchar minVal = UCHAR_MAX;
						for(int i=-1; i<=1; i++)
							if(outputMatrix.at<uchar>(row-1,col+i)<minVal)
								minVal = outputMatrix.at<uchar>(row-1,col+i);
						if(outputMatrix.at<uchar>(row,col-1)<minVal)
							minVal = outputMatrix.at<uchar>(row,col-1);
						outputMatrix.at<uchar>(row,col) = min(minVal+1,255);
					}
				}
			}
			//seconda scansione (basso->alto & destra->sinistra)
			for(int row=N-2; row>1; row--){
				for(int col=M-2; col>1; col--){
					int actualPixelValue = outputMatrix.at<uchar>(row,col);
					if(actualPixelValue!=0){
					uchar minVal = UCHAR_MAX;
						for(int i=-1; i<=1; i++)
							if(outputMatrix.at<uchar>(row+1,col+i)<minVal)
								minVal = outputMatrix.at<uchar>(row+1,col+i);
						if(outputMatrix.at<uchar>(row,col+1)<minVal)
							minVal = outputMatrix.at<uchar>(row,col+1);
						outputMatrix.at<uchar>(row,col) = min(min(minVal+1, actualPixelValue),255);
					}
				}
			}
			showNormalizedOutputMatrix();
			//imwrite("outputImage.png", outputMatrix);
			return outputMatrix;
		}
};



int main( int argc, char** argv ){
    const char* fileName = (argc>=2)? argv[1]:"./circle.jpg";
	
	Mat sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);
    
	if (sourceMatrix.empty()){
        cout<<"--- EMPTY IMAGE ---"<<endl;
        exit(EXIT_FAILURE);
    }
	
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", sourceMatrix);

	DistanceTransform transform = DistanceTransform(sourceMatrix);
	Mat outputMatrix = transform.applyTransform();
	
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", outputMatrix);
	
    waitKey(0);
    return 0;
}