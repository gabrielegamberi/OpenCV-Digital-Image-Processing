#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
//#define RADCOEF CV_PI/180
#define RADCOEF CV_PI/180

using namespace std;
using namespace cv;

int houghThreshold;

void HoughTransformLine(Mat, Mat&);
void calcXYCoord(double, double, Point&, Point&);

int main(int argc, char** argv) {

	if(argc!=3){
		cerr<< "Usage: ./<programname> <imagename.format> <threshold>" << endl;
		exit(EXIT_FAILURE);
	}
	Mat sourceMatrix = imread(argv[1], IMREAD_GRAYSCALE);
	houghThreshold = atoi(argv[2]);
	
	if(sourceMatrix.empty()){
		cerr<< "Image format not valid." << endl;
		exit(EXIT_FAILURE);
	}
	
	Mat blurredMatrix, edgeMatrix;
	Mat outputMatrix = sourceMatrix.clone();
	
	//Before calling Hough we have to apply some blurring to the image (dim: 5x5) to then use Canny Edge Detector
	GaussianBlur(sourceMatrix, blurredMatrix, Size(5,5), 1.4, 0.0);
	imshow("Gaussian Blur", blurredMatrix);

	Canny(blurredMatrix, edgeMatrix, 60, 150, 3); //(InMatrix, OutMatrix, Threshold #1 & 2 for hysteresis, dim. SobelKernel)
	imshow("Canny Edge", edgeMatrix);
	waitKey(0);
	
	//Applying Hough transform
	HoughTransformLine(edgeMatrix, outputMatrix);
	
	//Showing the result
	imshow("Hough Lines", outputMatrix);
	waitKey(0);
	return(0);

}

void HoughTransformLine(Mat edgeMatrix, Mat& outputMatrix) {
	int maxDistance = max(edgeMatrix.rows, edgeMatrix.cols)*sqrt(2);
	Mat votes = Mat::zeros(maxDistance, 180, CV_8UC1);	//initialize to 0
	
	double rho, theta;

	//iterate through all pixels
	for(int y=0; y<edgeMatrix.rows; y++){
		for(int x=0; x<edgeMatrix.cols; x++){
			//if the current pixel is an edge pixel
			if(edgeMatrix.at<uchar>(y,x) > 250){	
				//for all the possible lines for that point (0->180)
				for(theta=0; theta<180; theta++){
					//rho = xcos(theta in rad.) + ysin(theta in rad.)     -> convert theta in radiants
					rho = cvRound(x*cos(theta*RADCOEF) + y*sin(theta*RADCOEF));
					//If rho is < 0 add max_distance to it. if it is > than the max_distance, subtract from it.
					rho = rho<0? rho+maxDistance : rho>maxDistance? rho-maxDistance : rho;
					votes.at<uchar>(rho, theta)++;
				}
			}
		}
	}

	imshow("Votes", votes);
	waitKey(0);
	
	vector<pair<Point,Point>> lines;	//vector of lines (starting Point, ending Point)
	Point P1, P2;
	
	cvtColor(outputMatrix, outputMatrix, COLOR_GRAY2BGR); //useful to color the image with red lines
	
	//We've got to find the peak point that are greater than a specific threshold. If the i-th point meets the condition, 
	//we must translate the polar coordinate to the cartesian one, and create the relative points to display.
	for(int r=0; r<votes.rows; r++) {
		for(int t=0; t<180; t++) {
			if(votes.at<uchar>(r,t) >= houghThreshold) {
				rho = r;
				theta = t*RADCOEF;
				calcXYCoord(rho, theta, P1, P2);
				//push the points into the vector
				lines.push_back(make_pair(P1,P2));
			}		
		}	
	}
	//let's now draw the line
	for(int i=0; i<lines.size(); i++) {
		pair<Point, Point> coordinates = lines.at(i); //pick up a pair of coordinates
		line(outputMatrix, coordinates.first, coordinates.second, Scalar(0, 0, 255), 1, LINE_AA); //draw a line
	}
}

//We translate the polar coordinates into the cartesian ones.
void calcXYCoord(double rho, double theta, Point& P1, Point& P2) {
	int lineLength = 1000; //we can use the image size (#rows or #columns)
	//x0 and y0 are our starting points
	int x0 = cvRound(rho * cos(theta));
	int y0 = cvRound(rho * sin(theta));
	
	//let's build the points
	P1.x = cvRound(x0 - lineLength * (+sin(theta)));
	P1.y = cvRound(y0 - lineLength * (-cos(theta)));
	P2.x = cvRound(x0 + lineLength * (+sin(theta)));
	P2.y = cvRound(y0 + lineLength * (-cos(theta)));
	
}