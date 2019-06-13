
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
					yGrad+=(int)(paddedMatrix.at<uchar>(row+i,col+j)*horSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
					xGrad+=(int)(paddedMatrix.at<uchar>(row+i,col+j)*vertSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
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
			if(edgeDirection == vertical){ //orientazione edge verticale
				neighborPix1 = sobelMag.at<float>(row-1,col);
				neighborPix2 = sobelMag.at<float>(row+1,col);
			}else if(edgeDirection == pDiagonal){ //diagonale a 45°
				neighborPix1 = sobelMag.at<float>(row+1,col-1);
				neighborPix2 = sobelMag.at<float>(row-1,col+1);
			}else if(edgeDirection == horizontal){ //orizzontale
				neighborPix1 = sobelMag.at<float>(row,col-1);
				neighborPix2 = sobelMag.at<float>(row,col+1);
			}else{									//diagonale a -45°
				neighborPix1 = sobelMag.at<float>(row-1,col-1);
				neighborPix2 = sobelMag.at<float>(row+1,col+1);
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

#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <math.h>
using namespace std;
using namespace cv;

#define KERNEL_SIZE 3
#define PAD 1

int N, M;
int maxThreshold, minThreshold;
Mat paddedMatrix;

Mat horSobel = (Mat_<char>(KERNEL_SIZE, KERNEL_SIZE)<<-1,-2,-1,
													   0, 0, 0,
													   1, 2, 1
);

Mat vertSobel = (Mat_<char>(KERNEL_SIZE, KERNEL_SIZE)<<-1, 0, 1,
													   -2, 0, 2,
													   -1, 0, 1
);

enum direction{horizontal=1, vertical, pDiagonal, nDiagonal};

void showNormalizedMatrix(String title, const Mat matrix){
	Mat normalizedMatrix;
	normalize(matrix, normalizedMatrix, 0, 255, NORM_MINMAX, CV_8UC1);
	namedWindow(title, WINDOW_AUTOSIZE);
	imshow(title, normalizedMatrix);
}

void sobel(Mat& paddedMatrix, Mat& sobelX, Mat& sobelY, Mat& magnitude, Mat& orientation){
	int xGrad, yGrad;
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			xGrad = yGrad = 0;
			for(int i=-PAD; i<=PAD; i++){
				for(int j=-PAD; j<=PAD; j++){
					xGrad+=paddedMatrix.at<uchar>(row+i, col+j)*vertSobel.at<char>(i+PAD, j+PAD);
					yGrad+=paddedMatrix.at<uchar>(row+i, col+j)*horSobel.at<char>(i+PAD, j+PAD);
				}
			}
			sobelX.at<float>(row,col) = abs(xGrad);
			sobelY.at<float>(row,col) = abs(yGrad);

			float orientationValue = atan2(xGrad, yGrad)*180/CV_PI;
			orientationValue = (orientationValue<0)? orientationValue+180:orientationValue;

			if(orientationValue>=22.5 && orientationValue<=67.5)
				orientationValue = pDiagonal;
			else if(orientationValue>67.5 && orientationValue<=112.5)
				orientationValue = horizontal;
			else if(orientationValue>112.5 && orientationValue<=157.5)
				orientationValue = nDiagonal;
			else
				orientationValue = vertical;

			orientation.at<uchar>(row,col) = orientationValue;
			magnitude.at<float>(row,col) = sqrt(pow(sobelX.at<float>(row,col),2)+pow(sobelY.at<float>(row,col),2));

		}	
	}	
}


void nonMaximaSuppression(Mat& trueEdges, const Mat magnitude, const Mat orientation){
	float currMagnitude;
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			currMagnitude = magnitude.at<float>(row,col);
			int edgeDirection = orientation.at<uchar>(row,col);
			float neighborMag1, neighborMag2;
			if(edgeDirection==horizontal){
				neighborMag1 = magnitude.at<float>(row,col-1);
				neighborMag2 = magnitude.at<float>(row,col+1);
			}else if(edgeDirection==vertical){
				neighborMag1 = magnitude.at<float>(row-1,col);
				neighborMag2 = magnitude.at<float>(row+1,col);
			}else if(edgeDirection==pDiagonal){
				neighborMag1 = magnitude.at<float>(row-1,col+1);
				neighborMag2 = magnitude.at<float>(row+1,col-1);
			}else{
				neighborMag1 = magnitude.at<float>(row-1,col-1);
				neighborMag2 = magnitude.at<float>(row+1,col+1);
			}
			currMagnitude = (currMagnitude<max(neighborMag1,neighborMag2))? 0:currMagnitude;
			trueEdges.at<float>(row,col) = currMagnitude;
		}	
	}
}
void isteresis(Mat& trueEdges, Mat& almostEdges, Mat& finalEdges){
	finalEdges = Mat::zeros(trueEdges.size(), CV_8UC1);
	float currentPixelVal;
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			currentPixelVal = trueEdges.at<float>(row,col);
			trueEdges.at<float>(row,col) = (currentPixelVal>=maxThreshold)? currentPixelVal:0;
			almostEdges.at<float>(row,col) = (currentPixelVal>=minThreshold && currentPixelVal<maxThreshold)? currentPixelVal:0;
			finalEdges.at<uchar>(row,col) = (trueEdges.at<float>(row,col)!=0)? 255:0;
		}
	}
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			currentPixelVal = trueEdges.at<float>(row,col);
			if(currentPixelVal>0){
				for(int i=-PAD; i<=PAD; i++){
					for(int j=-PAD; j<=PAD; j++){
						float neighborPixelVal = almostEdges.at<float>(row+i,col+j);
						if((i|j)!=0 && neighborPixelVal>0){
							trueEdges.at<float>(row+i,col+j) = neighborPixelVal;
							finalEdges.at<uchar>(row+i,col+j) = 255;
						}
					}
				}
			}
		}
	}
}

void CannyEdgeDetection(const Mat paddedMatrix){
	Mat blurredMatrix, sobelX, sobelY, magnitude, orientation;
	GaussianBlur(paddedMatrix, blurredMatrix, Size(3,3), 1.4);
	
	sobelX = Mat(blurredMatrix.size(), CV_32FC1);
	sobelY = Mat(blurredMatrix.size(), CV_32FC1);
	magnitude = Mat(blurredMatrix.size(), CV_32FC1);
	orientation = Mat(blurredMatrix.size(), CV_8UC1);



	sobel(blurredMatrix, sobelX, sobelY, magnitude, orientation);

	Mat trueEdges, almostEdges, finalEdges;
	trueEdges = Mat::zeros(paddedMatrix.size(), CV_32FC1);
	almostEdges = Mat::zeros(paddedMatrix.size(), CV_32FC1);

	nonMaximaSuppression(trueEdges, magnitude, orientation);

	isteresis(trueEdges, almostEdges, finalEdges);

	showNormalizedMatrix("sobelX", sobelX);
	showNormalizedMatrix("sobelY", sobelY);
	showNormalizedMatrix("magnitude", magnitude);
	showNormalizedMatrix("true edges", trueEdges);
	showNormalizedMatrix("Final Edges", finalEdges);
	
}

void invokeMinCanny(int newMinThreshold, void*){
	minThreshold = newMinThreshold;
	CannyEdgeDetection(paddedMatrix);
}
void invokeMaxCanny(int newMaxThreshold, void*){
	maxThreshold = newMaxThreshold;
	CannyEdgeDetection(paddedMatrix);
}

int main(int argc, char** argv ){
	if(argc!=4){
		cerr<<"--- ERROR --- current usage: <exe> <file.ext> <minThreshold> <maxThreshold>"<<endl;
		exit(EXIT_FAILURE);
	}
   	const char* fileName = argv[1];
	Mat sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);
	if(sourceMatrix.empty()){
		cerr<<"-- Matrix is empty --"<<endl;
		exit(EXIT_FAILURE);
	}
	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", sourceMatrix);
	
	paddedMatrix = Mat::zeros(sourceMatrix.rows+2*PAD, sourceMatrix.cols+2*PAD, sourceMatrix.type());
	for(int i=PAD; i<paddedMatrix.rows-PAD; i++)
		for(int j=PAD; j<paddedMatrix.cols-PAD; j++)
			paddedMatrix.at<uchar>(i,j) = sourceMatrix.at<uchar>(i-PAD,j-PAD);
	
	N = paddedMatrix.rows;
	M = paddedMatrix.cols;
	minThreshold = atoi(argv[2]);
	maxThreshold = atoi(argv[3]);

	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", sourceMatrix);
	
	int thresholdPassed;
	createTrackbar("Min Threshold", "input", &thresholdPassed, 255, invokeMinCanny);
	createTrackbar("Max Threshold", "input", &thresholdPassed, 255, invokeMaxCanny);
	waitKey(0);
	//CannyEdgeDetection(paddedMatrix);
	//Mat noPaddedMatrix;
	//paddedMatrix(Rect(Point(PAD, paddedMatrix.cols-PAD), Point(paddedMatrix.rows-PAD, PAD))).copyTo(noPaddedMatrix);
	//namedWindow("output", WINDOW_AUTOSIZE);
	//imshow("output", noPaddedMatrix);
	
	return 0;
}

*/