
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;
#define SIZE 5
#define PAD floor(SIZE/2)
#define SOBEL_SIZE 3
#define SOBEL_PAD floor(SOBEL_SIZE/2)

enum direction {vertical=1, pDiagonal, horizontal, nDiagonal};

Mat kernel(SIZE,SIZE,CV_32FC1,Scalar(0));
/*Mat kernel = Mat::ones(LEN,LEN,CV_8U);
Mat mask = (Mat_<char>(LEN,LEN)<<1,1,1,
                                 1,1,1,
                                 1,1,1
);
*/

Mat horSobel = (Mat_<char>(SOBEL_SIZE,SOBEL_SIZE)<<-1,-2,-1,
									   				 0, 0, 0,
									   				 1, 2, 1
);
Mat vertSobel = (Mat_<char>(SOBEL_SIZE,SOBEL_SIZE)<<-1, 0, 1,
									    			-2, 0, 2,
									    			-1, 0, 1
);

Mat paddedMatrix;
int N, M;
int minThreshold, maxThreshold;

/*void buildGaussianFilterWithReflection(){
	int variance = 1;
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
*/

void showNormalizedMatrix(String label, const Mat matrix){
	Mat normalizedMatrix = Mat::zeros(matrix.size(), matrix.type());
	cv::normalize(matrix, normalizedMatrix, 0, 255, NORM_MINMAX, CV_8UC1);
	namedWindow(label, WINDOW_AUTOSIZE);
	imshow(label, normalizedMatrix);
}

void buildGaussianFilter(){
	float variance = 4;
	int rowShift, colShift, B;
	float minTerm;
	for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			kernel.at<float>(i,j) = exp(-(pow(abs(i-PAD),2)+pow(abs(j-PAD),2))/(2*variance));
	minTerm = kernel.at<float>(kernel.rows-1,kernel.cols-1);

	for(int i=0; i<kernel.rows; i++){
		for(int j=0; j<kernel.cols; j++){
			B = round(kernel.at<float>(i,j)/minTerm);
			kernel.at<float>(i,j) = B;	
		}
	}
}

void gaussianFilter(Mat& inoutPaddedMatrix){
	int denom=0;
	Mat temp = inoutPaddedMatrix.clone();
	
	buildGaussianFilter();

	for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			denom+=kernel.at<float>(i,j);

	int maskedValue[SIZE*SIZE];
	for(int row=PAD; row<temp.rows-PAD; row++){
		for(int col=PAD; col<temp.cols-PAD; col++){
			int mask_index = 0;
			for(int i=-PAD; i<=PAD; i++)
				for(int j=-PAD; j<=PAD; j++)
					maskedValue[mask_index++] = temp.at<uchar>(row+i,col+j)*kernel.at<float>(i+PAD, j+PAD);
			int sum=0;
			for(int i=0; i<SIZE*SIZE; i++)
				sum+=maskedValue[i];
			sum=sum/denom;
			inoutPaddedMatrix.at<uchar>(row,col) = sum;
		}
	}
}

void sobel(const Mat inputMatrix, Mat& sobelGx, Mat& sobelGy, Mat& sobelMag, Mat &orMatrix){
	int yGrad, xGrad;
	int N = paddedMatrix.rows;
	int M = paddedMatrix.cols;

	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			yGrad = xGrad = 0;
			for(int i=-SOBEL_PAD; i<=SOBEL_PAD; i++){
				for(int j=-SOBEL_PAD; j<=SOBEL_PAD; j++){
					yGrad+=(int)(paddedMatrix.at<uchar>(row+i,col+j)*vertSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
					xGrad+=(int)(paddedMatrix.at<uchar>(row+i,col+j)*horSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
				}
			}

			float orVal = atan2(yGrad,xGrad)*180/M_PI;	//direzione del gradiente
			orVal = (orVal<0)? orVal+180:orVal;

			if(orVal>=22.5 && orVal<=67.5)
				orVal = pDiagonal;	//orientazione edge diagonale a 45°
			else if(orVal>=67.5 && orVal<=112.5)
				orVal = horizontal; //orientazione edge orizzontale
			else if(orVal>=112.5 && orVal<=157.5)
				orVal = nDiagonal; //orientazione edge diagonale a -45°
			else
				orVal = vertical; //orientazione edge verticale
			
			orMatrix.at<uchar>(row,col) = orVal;	//assegnazione della direzione del gradiente
			sobelGx.at<float>(row,col) = abs(xGrad);
			sobelGy.at<float>(row,col) = abs(yGrad);
			sobelMag.at<float>(row,col) = sqrt(pow(xGrad,2)+pow(yGrad,2));
		}
	}
	//showNormalizedMatrix("XGrad", sobelGx);
	//showNormalizedMatrix("YGrad", sobelGy);
	//showNormalizedMatrix("SobelMag", sobelMag);
}
void nonMaximaSuppression(Mat& trueEdges, const Mat sobelMag, const Mat orMatrix){
	float currentPixel;
	//Inizio soppressione dei non massimi
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			currentPixel = sobelMag.at<float>(row,col);
			int edgeDirection = orMatrix.at<float>(row,col);
			int neighborPix1, neighborPix2;
			if(edgeDirection == vertical){ //orientazione edge verticale -> prendo i pixel ortogonali (dir. del gradiente = orizzontale)
				neighborPix1 = sobelMag.at<float>(row,col-1);
				neighborPix2 = sobelMag.at<float>(row,col+1);
			}else if(edgeDirection == pDiagonal){ //diagonale a 45°
				neighborPix1 = sobelMag.at<float>(row-1,col-1);
				neighborPix2 = sobelMag.at<float>(row+1,col+1);
			}else if(edgeDirection == horizontal){ //orizzontale
				neighborPix1 = sobelMag.at<float>(row-1,col);
				neighborPix2 = sobelMag.at<float>(row+1,col);
			}else{									//diagonale a -45°
				neighborPix1 = sobelMag.at<float>(row+1,col-1);
				neighborPix2 = sobelMag.at<float>(row-1,col+1);
			}
			currentPixel = (currentPixel<max(neighborPix1,neighborPix2))? 0:currentPixel;	//non maxima suppression
			trueEdges.at<float>(row,col) = currentPixel; //(currentPixel>maxVal)? currentPixel:0; 
		}
	}
}

void secondChance(Mat& trueEdges, Mat& almostEdges, int row, int col){
	for(int i=-1; i<=1; i++){
		for(int j=-1; j<=1; j++){
			int neighborPix = almostEdges.at<uchar>(row+i,col+j);
			if(i!=0 && j!=0 && neighborPix!=0){ //se non sei il pixel centrale e sei diverso da 0
				trueEdges.at<uchar>(row+i,col+j) = neighborPix;
				almostEdges.at<uchar>(row+i,col+j) = 0;
				secondChance(trueEdges, almostEdges, row+i, col+j);
			}
		}
	}
	return;
}

void isteresi(Mat& trueEdges, Mat& almostEdges){
	Mat output = Mat::zeros(trueEdges.size(), CV_8UC1);
	float currentPixel;
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			currentPixel = trueEdges.at<float>(row,col);
			//isteresi (se supero la soglia massima, allora sicuramente il pixel corrente è un edge, altrimenti, può essere "quasi un edge")
			trueEdges.at<float>(row,col) = (currentPixel>=maxThreshold)? currentPixel:0; 
			almostEdges.at<float>(row,col) = (currentPixel>minThreshold && currentPixel<maxThreshold)? currentPixel:0;
			output.at<uchar>(row,col) = (trueEdges.at<float>(row,col)!=0)? 255:0;
		}
	}
	//Processo di "seconda chance" ai "quasi edge"
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			currentPixel = trueEdges.at<float>(row,col);
			if(currentPixel>0){
				//secondChance(trueEdges, almostEdges, row, col);
				for(int i=-1; i<=1; i++){	//controllo la 8-connettività
					for(int j=-1; j<=1; j++){
						float neighborPix = almostEdges.at<float>(row+i,col+j);
						if(i!=0 && j!=0 && neighborPix!=0){ //se non sei il pixel centrale e sei diverso da 0
							trueEdges.at<float>(row+i,col+j) = neighborPix;
							output.at<uchar>(row+i,col+j) = 255;
							//secondChance(trueEdges, almostEdges, row+i, col+j);
							//almostEdges.at<uchar>(row+i,col+j) = 0;
						}
					}
				}
			}
		}
	}
	showNormalizedMatrix("Final Edges", output);
}


void Canny(const Mat inputMatrix){
	//int maxVal = 0;
	Mat paddedMatrix = inputMatrix.clone();
	Mat sobelGx, sobelGy, sobelMag, orMatrix;
	sobelGx = Mat(paddedMatrix.size(), CV_32FC1);
	sobelGy = Mat(paddedMatrix.size(), CV_32FC1);
	sobelMag = Mat(paddedMatrix.size(), CV_32FC1);
	orMatrix = Mat(paddedMatrix.size(), paddedMatrix.type());

	//gaussianFilter(paddedMatrix);
	
	//showNormalizedMatrix("gaussianMat", paddedMatrix);
	
	sobel(paddedMatrix, sobelGx, sobelGy, sobelMag, orMatrix);
	
	cout<<"--THRESHOLD--"<<endl;
	cout<<"MAX: "<<maxThreshold<<endl;
	cout<<"MIN: "<<minThreshold<<endl;
	
	//Mat trueEdges = sobelMag.clone();
	Mat trueEdges = Mat::zeros(sobelMag.size(), CV_32FC1);
	Mat almostEdges = Mat::zeros(sobelMag.size(), CV_32FC1);

	nonMaximaSuppression(trueEdges, sobelMag, orMatrix);

	isteresi(trueEdges, almostEdges);

	showNormalizedMatrix("Almost Edges", almostEdges);
	showNormalizedMatrix("True Edges", trueEdges);
}

void invokeMinCanny(int selectedThreshold, void*){
	minThreshold = selectedThreshold;
	Canny(paddedMatrix);
}
void invokeMaxCanny(int selectedThreshold, void*){
	maxThreshold = selectedThreshold;
	Canny(paddedMatrix);
}



int main(int argc, char** argv )
{
   	const char* fileName = (argc>=2)? argv[1]:"./lena.jpg";
	Mat inputMatrix = imread(fileName, IMREAD_GRAYSCALE);
	if(inputMatrix.empty()){
		cerr<<"-- Matrix is empty --"<<endl;
		exit(EXIT_FAILURE);
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", inputMatrix);

	int threshold;
  	createTrackbar("Min Threshold:", "input", &threshold, 255, invokeMinCanny);
	createTrackbar("Max Threshold:", "input", &threshold, 255, invokeMaxCanny);

	paddedMatrix = Mat::zeros(inputMatrix.rows+2*PAD, inputMatrix.cols+2*PAD, inputMatrix.type());
	for(int i=PAD; i<paddedMatrix.rows-PAD; i++)
		for(int j=PAD; j<paddedMatrix.cols-PAD; j++)
			paddedMatrix.at<uchar>(i,j) = inputMatrix.at<uchar>(i-PAD, j-PAD);
	N = paddedMatrix.rows;
	M = paddedMatrix.cols;

	//Mat noPaddedMatrix;
	//paddedMatrix(Rect(Point(PAD, paddedMatrix.cols-PAD), Point(paddedMatrix.rows-PAD, PAD))).copyTo(noPaddedMatrix);
	//namedWindow("output", WINDOW_AUTOSIZE);
	//imshow("output", noPaddedMatrix);
	
	waitKey(0);
	return 0;
}
  

/*
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

using namespace cv;

/// Global variables

Mat src, src_gray;
Mat dst, detected_edges;

int edgeThresh = 1;
int lowThreshold;
int const max_lowThreshold = 100;
int ratio = 3;
int kernel_size = 3;
String window_name = "Edge Map";


void CannyThreshold(int, void*)
{
  /// Reduce noise with a kernel 3x3
  blur( src_gray, detected_edges, Size(3,3) );

  /// Canny detector
  Canny( detected_edges, detected_edges, lowThreshold, lowThreshold*ratio, kernel_size );

  /// Using Canny's output as a mask, we display our result
  dst = Scalar::all(0);

  src.copyTo( dst, detected_edges);
  imshow( window_name, dst );
 }


int main( int argc, char** argv )
{
  /// Load an image
  src = imread( "./lena.jpg" );

  if( !src.data )
  { return -1; }

  /// Create a matrix of the same type and size as src (for dst)
  dst.create( src.size(), src.type() );

  /// Convert the image to grayscale
  cvtColor( src, src_gray, CV_BGR2GRAY );

  /// Create a window
  namedWindow( window_name, CV_WINDOW_AUTOSIZE );

  /// Create a Trackbar for user to enter threshold
  createTrackbar( "Min Threshold:", window_name, &lowThreshold, max_lowThreshold, CannyThreshold );

  /// Show the image
  CannyThreshold(0, 0);

  /// Wait until user exit program by pressing a key
  waitKey(0);

  return 0;
  }
  */