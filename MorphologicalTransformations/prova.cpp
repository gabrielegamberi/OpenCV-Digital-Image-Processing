#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <limits.h>
#include <tuple>

using namespace cv;
using namespace std;

#define KERNEL_SIZE 3
#define PAD 1

		
class MorphTransform{
	private:
		Mat structElem;
		int connectivity;
		Mat sourceMatrix;
		Mat outputMatrix;
		int N;
		int M;
	
	bool isOutsideTheMatrix(int row, int col){
		if(row<0 || row>=N || col<0 || col>=M)
			return true;
		return false;
	}

	void threshold(const Mat source, Mat& outputMatrix){
		float min, max;
		min = 255;
		max = 0;
		for(int row=0; row<N; row++){
			for(int col=0; col<M; col++){
				if(source.at<uchar>(row,col)>max)
					max = source.at<uchar>(row,col);
				if(source.at<uchar>(row,col)<min)
					min = source.at<uchar>(row,col);
			}
		}
		int median = floor((max+min)/2);
		for(int row=0; row<N; row++){
			for(int col=0; col<M; col++){
				if(source.at<uchar>(row,col)>median)
					outputMatrix.at<uchar>(row,col) = 255;
				if(source.at<uchar>(row,col)<median)
					outputMatrix.at<uchar>(row,col) = 0;
			}
		}
	}

	int calcSumSurroundings(const Mat temp, int row, int col){
		int accum = 0;
		for(int i=-PAD; i<=PAD; i++){
			for(int j=-PAD; j<=PAD; j++){
				if(isOutsideTheMatrix(row+i,col+j))
					continue;
				accum+=temp.at<uchar>(row+i,col+j)*structElem.at<uchar>(i+PAD,j+PAD);
			}
		}
		return accum;
	}

	public:
		MorphTransform(const Mat source):sourceMatrix(source){
			structElem = (Mat_<uchar>(KERNEL_SIZE,KERNEL_SIZE)<<0,1,0,
																1,1,1,
																0,1,0
			);
			connectivity = 4;
			N = source.rows;
			M = source.cols;
			resetOutput();
		}

		void erode(int nTimes=1){
			Mat temp;
			while(nTimes-->0){
				temp = outputMatrix.clone();
				for(int row=0; row<N; row++){
					for(int col=0; col<M; col++){
						if(temp.at<uchar>(row,col)==255){
							int sum = calcSumSurroundings(temp, row,col);
							if(sum < 255*(connectivity+1))
								outputMatrix.at<uchar>(row,col) = 0;

						}
					}
				}
			}
		}
		void dilate(int nTimes=1){
			Mat temp;
			while(nTimes-->0){
				temp = outputMatrix.clone();
				for(int row=0; row<N; row++){
					for(int col=0; col<M; col++){
						int sum = calcSumSurroundings(temp,row,col);
						if(sum>0)
							outputMatrix.at<uchar>(row,col) = 255;
					}
				}
			}
		}

		void open(int nTimes=1){
			while(nTimes-->0){
				erode();
				dilate();
			}
		}

		void close(int nTimes=1){
			while(nTimes-->0){
				dilate();				
				erode();
			}
		}

		void distanceTransform(){
			//definisco la matrice dove accumulerò le erosioni
			//float in quanto ci saranno valori molto alti
			Mat temp = outputMatrix.clone();
			Mat R = Mat::zeros(temp.size(), CV_32FC1);   
			for(int i=PAD; i<temp.rows-PAD; i++){
				for(int j=PAD; j<temp.cols-PAD; j++){
					R.at<float>(i,j) = temp.at<uchar>(i,j);
				}
			}

			for(int index=0; index<256; index++){
				for(int i=PAD; i<temp.rows-PAD; i++){
					for(int j=PAD; j<temp.cols-PAD; j++){
						R.at<float>(i,j) += temp.at<uchar>(i, j);
					}
				}
				erode();
				temp = getResult();
			}

			//normalizzo i valori per farli rientrare nel range
			//NORM_MINMAX permette di normalizzare i valori tra 
			//min -> 0 e max -> 255
			normalize(R, outputMatrix, 0, 255, NORM_MINMAX, CV_8U);
		}

		inline Mat& getResult(){return outputMatrix;}
		void resetOutput(){
			outputMatrix = Mat(sourceMatrix.size(), sourceMatrix.type());
			threshold(sourceMatrix, outputMatrix);
		}
};


int main( int argc, char** argv ){
    const char* fileName = (argc>=2)? argv[1]:"./letter.png";
	
	Mat sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);

	if (sourceMatrix.empty()){
        cout<<"--- EMPTY IMAGE ---"<<endl;
        exit(EXIT_FAILURE);
    }
	
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", sourceMatrix);

	MorphTransform transform(sourceMatrix);
	Mat outputMatrix;
	transform.erode(10);					//EROSION
	outputMatrix = transform.getResult();
	transform.resetOutput();	
	namedWindow("erode", WINDOW_AUTOSIZE);
	imshow("erode",outputMatrix);

	
	transform.dilate(15);						//DILATION
	outputMatrix = transform.getResult();
	transform.resetOutput();
	namedWindow("dilation", WINDOW_AUTOSIZE);
	imshow("dilation",outputMatrix);
	
	transform.close(20);							//OPENING
	outputMatrix = transform.getResult();
	transform.resetOutput();
	namedWindow("closing", WINDOW_AUTOSIZE);
	imshow("closing",outputMatrix);

	transform.open(20);							//OPENING
	outputMatrix = transform.getResult();
	transform.resetOutput();
	namedWindow("opening", WINDOW_AUTOSIZE);
	imshow("opening",outputMatrix);
	
	transform.distanceTransform();	
	outputMatrix = transform.getResult();		
	transform.resetOutput();
	namedWindow("Distance Transform", WINDOW_AUTOSIZE);
	imshow("Distance Transform",outputMatrix);

	

    waitKey(0);
    return 0;
}