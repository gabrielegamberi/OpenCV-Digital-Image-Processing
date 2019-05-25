//SELEZIONE MANUALE DEL RANGE TRAMITE RIGA DI COMANDO 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <limits.h>
#include <list>
#include <time.h>

using namespace std;
using namespace cv;

class Region{
	private:
		Point seed;
		list<Point> points;

	public:
		static Mat sourceMatrix;
		static Mat outputMatrix;
		static Mat visitedMatrix;
		static float eucThreshold;
		
		Region(){}

		Region(Point seed){
			this->seed = seed;
			addPoint(seed);
		}
		inline int getRegionSize(){return points.size();}
		
		static bool isInsideRange(Point point){
			if(point.x<0 || point.y<0 || point.x>=sourceMatrix.cols || point.y>=sourceMatrix.rows)
				return false;
			return true;
		}

		static void init(char *fileName, float threshold){
			srand(time(NULL));
			sourceMatrix = imread(fileName, IMREAD_COLOR);	
			cv::resize(sourceMatrix, sourceMatrix, cv::Size(), 0.5, 0.5);
			blur(sourceMatrix, sourceMatrix, Size(3,3), Point(-1,-1));
			outputMatrix = Mat::zeros(sourceMatrix.size(), sourceMatrix.type()); //sourceMatrix.clone();
			visitedMatrix = Mat::zeros(sourceMatrix.size(), CV_8UC1);
			eucThreshold = threshold;
		}

		bool isDistanceOk(Point point, Point neighbor){
			Vec3b colorP = sourceMatrix.at<Vec3b>(point);
			Vec3b colorN = sourceMatrix.at<Vec3b>(neighbor);
			float eucDist = sqrt(pow(colorP[0]-colorN[0],2)+pow(colorP[1]-colorN[1],2)+pow(colorP[2]-colorN[2],2));
			if(eucDist<eucThreshold)
				return true;
			return false;
		}

		void addPoint(Point newPoint){
			points.push_front(newPoint);
			visitedMatrix.at<uchar>(newPoint) = 1;
		}

		void grow(){
			list<Point> processList;
			processList.push_front(seed);
			while(!processList.empty()){
				Point curPoint = processList.front(); 
				processList.pop_front();
				for(int i=-1; i<=1; i++){
					for(int j=-1; j<=1; j++){
						Point neighPoint = Point(curPoint.x+i, curPoint.y+j);
						if((i|j)!=0 && isInsideRange(neighPoint)){
							if(visitedMatrix.at<uchar>(neighPoint) == 0){
								if(isDistanceOk(curPoint, neighPoint)){
									addPoint(neighPoint);
									processList.push_front(neighPoint);
								}
							}
						}
					}
				}
			}
		}
		void recursiveGrow(Point curPoint){
			addPoint(curPoint);
			for(int i=-1; i<=1; i++){
				for(int j=-1; j<=1; j++){
					Point neighPoint = Point(curPoint.x+i, curPoint.y+j);
					if((i|j)!=0 && isInsideRange(neighPoint)){
						if(visitedMatrix.at<uchar>(neighPoint) == 0){
							//float aftVariance = pushInsideRegion(neighPoint);
							if(isDistanceOk(curPoint, neighPoint)){
								recursiveGrow(neighPoint);
							}
						}
					}
				}
			}
		}
		
		Vec3b getMeanColor(){
			float R, G, B;
			R = G = B = 0;
			list<Point>::iterator it;
			for(it=points.begin(); it!=points.end(); it++){
				Point point = (*it);
				B+=sourceMatrix.at<Vec3b>(point)[0];
				G+=sourceMatrix.at<Vec3b>(point)[1];
				R+=sourceMatrix.at<Vec3b>(point)[2];
			}
			B = cvRound(B/points.size());
			G = cvRound(G/points.size());
			R = cvRound(R/points.size());
			return Vec3b(B,G,R);
		}

		void color(){
			Vec3b color = getMeanColor();
			list<Point>::iterator it;
			for(it=points.begin(); it!=points.end(); it++){
				outputMatrix.at<Vec3b>(*it) = color;
			}
		}

};

float Region::eucThreshold = 0;
Mat Region::sourceMatrix;
Mat Region::outputMatrix;
Mat Region::visitedMatrix;



/*Region Growing from a standard seed (iterative version)*/
void regionGrowing(int nSeeds){
	double minVal = 0;
	Point newSeed(rand()%Region::sourceMatrix.cols, rand()%Region::sourceMatrix.rows);
	do{
		Region newRegion(newSeed);
		newRegion.grow();
		if(newRegion.getRegionSize()>0)
			newRegion.color();
		if(nSeeds>0){
			newSeed = Point(rand()%Region::sourceMatrix.cols, rand()%Region::sourceMatrix.rows);
			nSeeds--;
		}else
			minMaxLoc(Region::visitedMatrix, &minVal, nullptr, &newSeed, nullptr);
	}while(minVal==0);
	imwrite("region.png", Region::outputMatrix);
}

/*Region Growing from a standard seed (iterative version)*/
void recursiveRegionGrowing(int nSeeds){
	double minVal = 0;
	Point newSeed(rand()%Region::sourceMatrix.cols, rand()%Region::sourceMatrix.rows);
	do{
		Region newRegion;
		newRegion.recursiveGrow(newSeed);
		if(newRegion.getRegionSize()>0)
			newRegion.color();
		if(nSeeds>0){
			newSeed = Point(rand()%Region::sourceMatrix.cols, rand()%Region::sourceMatrix.rows);
			nSeeds--;
		}else
			minMaxLoc(Region::visitedMatrix, &minVal, nullptr, &newSeed, nullptr);
	}while(minVal==0);
	imwrite("region.png", Region::outputMatrix);
}


int main(int argc, char** argv) {
	if(argc!=3){
		cerr<< "Usage: ./<programname> <imagename.format> <threshold>" << endl;
		exit(EXIT_FAILURE);
	}
	
	Region::init(argv[1], atoi(argv[2]));
	
	if(Region::sourceMatrix.empty()){
		cerr<< "Image format not valid." << endl;
		exit(EXIT_FAILURE);
	}
	
	namedWindow("source", WINDOW_AUTOSIZE);
	imshow("source", Region::sourceMatrix);
	waitKey(0);
	//recursiveRegionGrowing(10);
	regionGrowing(10);

	return 0;
}
