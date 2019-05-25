#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <limits.h>
#include <tuple>

using namespace cv;
using namespace std;

#define KERNEL_SIZE 5
#define PAD(x) floor(x/2)

		
class MorphTransform{
	private:
		Mat structElem;
		Mat sourceMatrix;
		Mat outputMatrix;
		int N;
		int M;
		pair<int,int> centerCoord; //x,y coordinates of structElem's reference
		int pad;
	
	bool isOutsideTheMatrix(int row, int col){
		if(row<0 || row>=N || col<0 || col>=M)
			return true;
		return false;
	}

	public:
		MorphTransform(const Mat elem, const Mat source):structElem(elem), sourceMatrix(source){
			outputMatrix = source.clone();
			N = source.rows;
			M = source.cols;
			pad = PAD(structElem.rows);
			centerCoord = make_pair(0,0);	//the center is supposed to be one
		}

		void erode(int nTimes=1){
			Mat temp;
			while(nTimes-->0){
				temp = outputMatrix.clone();
				for(int row=0; row<N; row++){
					for(int col=0; col<M; col++){
						if(temp.at<uchar>(row,col)==255){
							bool isEroded = false;
							for(int i=-pad; i<=pad; i++){
								for(int j=-pad; j<=pad; j++){
									if(isOutsideTheMatrix(row+i,col+j))
										continue;
									if(structElem.at<uchar>(i+pad,j+pad)==1 && temp.at<uchar>(row+i,col+j)==0 && 
									!isOutsideTheMatrix(row+centerCoord.first,col+centerCoord.second)){
										outputMatrix.at<uchar>(row+centerCoord.first,col+centerCoord.second) = 0;
										isEroded = true;
										break;
									}
								}
								if(isEroded)
									break;
							}
						}
					}
				}
			}
		}
		void dilate(int nTimes=1){
			Mat temp;
			while(nTimes-->0){
				temp = outputMatrix.clone();
				for(int row=0; row<N; row++)
					for(int col=0; col<M; col++){
						if(!isOutsideTheMatrix(row+centerCoord.first,col+centerCoord.second) &&
							temp.at<uchar>(row+centerCoord.first,col+centerCoord.second)==255)//if the actual pixel has to be dilated
								for(int i=-pad; i<=pad; i++) //I replicate the whole structure (where the ones are set)
									for(int j=-pad; j<=pad; j++)
										if(structElem.at<uchar>(i+pad,j+pad)==1 && !isOutsideTheMatrix(row+i,col+j))
											outputMatrix.at<uchar>(row+i,col+j) = 255;
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

		inline Mat& getResult(){return outputMatrix;}
		void reset(){outputMatrix = sourceMatrix.clone();}
};


int main( int argc, char** argv ){
    const char* fileName = (argc>=2)? argv[1]:"./letter.png";
	
	Mat sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);
    Mat structElem = Mat(KERNEL_SIZE,KERNEL_SIZE,CV_8UC1,Scalar(1)); //8-connettività

	if (sourceMatrix.empty()){
        cout<<"--- EMPTY IMAGE ---"<<endl;
        exit(EXIT_FAILURE);
    }
	
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", sourceMatrix);

	MorphTransform transform(structElem, sourceMatrix);
	Mat outputMatrix;
	transform.erode();					//EROSION
	outputMatrix = transform.getResult();
	transform.reset();	
	namedWindow("erode", WINDOW_AUTOSIZE);
	imshow("erode",outputMatrix);

	transform.dilate();						//DILATION
	outputMatrix = transform.getResult();
	transform.reset();
	namedWindow("dilation", WINDOW_AUTOSIZE);
	imshow("dilation",outputMatrix);
	
	/*
	transform.open();						//OPENING
	outputMatrix = transform.getResult();
	transform.reset();
	namedWindow("opening", WINDOW_AUTOSIZE);
	imshow("opening",outputMatrix);
	
	transform.close();						//CLOSING
	outputMatrix = transform.getResult();
	transform.reset();
	namedWindow("closing", WINDOW_AUTOSIZE);
	imshow("closing",outputMatrix);
	*/

    waitKey(0);
    return 0;
}