//SELEZIONE MANUALE DEL RANGE TRAMITE RIGA DI COMANDO 

#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <string>
#include <limits.h>
#include <list>
#include <time.h>

//#define RADCOEF CV_PI/180
#define RADCOEF CV_PI/180

using namespace std;
using namespace cv;

class Region{
	private:
		Point seed;
		float mean;
		float variance;
		list<Point> points;

	public:
		static Mat sourceMatrix;
		static Mat outputMatrix;
		static Mat visitedMatrix;
		static float varThreshold;
		
		Region(){}

		Region(Point seed){
			this->seed = seed;
			//pushInsideRegion(seed);
			addPoint(seed);
		}

		static int getPixelVal(Point point){
			return sourceMatrix.at<uchar>(point);
		}

		inline int getRegionSize(){return points.size();}
		
		static bool isInsideRange(Point point){
			if(point.x<0 || point.y<0 || point.x>sourceMatrix.cols || point.y>sourceMatrix.rows)
				return false;
			return true;
		}

		static void init(char *fileName, float threshold){
			srand(time(NULL));
			sourceMatrix = imread(fileName, IMREAD_GRAYSCALE);	//cv::resize(sourceMatrix, sourceMatrix, cv::Size(), 1, 1);
			blur(sourceMatrix, sourceMatrix, Size(3,3), Point(-1,-1));
			outputMatrix = Mat::zeros(sourceMatrix.size(), sourceMatrix.type()); //sourceMatrix.clone();
			cvtColor(outputMatrix, outputMatrix, COLOR_GRAY2BGR);
			visitedMatrix = Mat::zeros(sourceMatrix.size(), sourceMatrix.type());
			varThreshold = threshold;
		}

		double testVariance(Point point){
			int tempSize = points.size();
			double tempVar = variance;
			tempSize++;
			tempVar+=(tempSize-1)*pow(getPixelVal(point)-mean,2)/tempSize;
			return tempVar/tempSize;
		}

		void addPoint(Point newPoint){
			points.push_front(newPoint);
			if(points.size() == 1){
				mean = getPixelVal(newPoint);
				variance = 0;
			}else{
				variance +=(points.size()-1)*pow(getPixelVal(newPoint)-mean,2)/points.size();
				mean+=(getPixelVal(newPoint)-mean)/points.size();
			}
		}

		void grow(){
			list<Point> processList;
			processList.push_front(seed);
			while(!processList.empty()){
				Point curPoint = processList.front(); 
				processList.pop_front();
				visitedMatrix.at<uchar>(curPoint) = 1;
				for(int i=-1; i<=1; i++){
					for(int j=-1; j<=1; j++){
						Point neighPoint = Point(curPoint.x+i, curPoint.y+j);
						if((i|j)!=0 && isInsideRange(neighPoint)){
							if(visitedMatrix.at<uchar>(neighPoint) == 0){
								//float aftVariance = pushInsideRegion(neighPoint);
								float aftVariance = testVariance(neighPoint);
								if(aftVariance<varThreshold){
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
			visitedMatrix.at<uchar>(curPoint) = 1;
			for(int i=-1; i<=1; i++){
				for(int j=-1; j<=1; j++){
					Point neighPoint = Point(curPoint.x+i, curPoint.y+j);
					if((i|j)!=0 && isInsideRange(neighPoint)){
						if(visitedMatrix.at<uchar>(neighPoint) == 0){
							//float aftVariance = pushInsideRegion(neighPoint);
							float aftVariance = testVariance(neighPoint);
							if(aftVariance<varThreshold){
								recursiveGrow(neighPoint);
							}
						}
					}
				}
			}
		}

		void color(){
			int R = (rand()%255)+1;
			int G = (rand()%255)+1;
			int B = (rand()%255)+1;
			list<Point>::iterator it;
			for(it=points.begin(); it!=points.end(); it++){
				Point point = (*it);
				outputMatrix.at<Vec3b>(point) = Vec3b(B,G,R);
			}
		}

};

float Region::varThreshold = 0;
Mat Region::sourceMatrix;
Mat Region::outputMatrix;
Mat Region::visitedMatrix;



/*Region Growing from a standard seed (iterative version)*/
void regionGrowing(){
	double minVal = 0;
	Point newSeed(rand()%Region::sourceMatrix.cols, rand()%Region::sourceMatrix.rows);
	do{
		Region newRegion(newSeed);
		newRegion.grow();
		if(newRegion.getRegionSize()>10)
			newRegion.color();
		minMaxLoc(Region::visitedMatrix, &minVal, nullptr, &newSeed, nullptr);
	}while(minVal==0);
	imwrite("region.png", Region::outputMatrix);
}

/*Region Growing from a standard seed (iterative version)*/
void recursiveRegionGrowing(){
	double minVal = 0;
	Point newSeed(rand()%Region::sourceMatrix.cols, rand()%Region::sourceMatrix.rows);
	do{
		Region newRegion;
		newRegion.recursiveGrow(newSeed);
		if(newRegion.getRegionSize()>10)
			newRegion.color();
		minMaxLoc(Region::visitedMatrix, &minVal, nullptr, &newSeed, nullptr);
	}while(minVal==0);
	imwrite("region.png", Region::outputMatrix);
}

/*Region Growing with a scan: I must specify the number of seeds per row*/
void regionGrowingScan(int nSeeds, int padFromEdges){
	int spaceX = cvRound((Region::sourceMatrix.cols-padFromEdges)/nSeeds);
	int spaceY = cvRound((Region::sourceMatrix.rows-padFromEdges)/nSeeds);
	for(int col=spaceY; col<=Region::sourceMatrix.rows-padFromEdges; col=col+spaceY){
		for(int i=1; i<=nSeeds; i++){
			Region newRegion(Point(i*spaceX,col));
			newRegion.grow();
			if(newRegion.getRegionSize()>10)
				newRegion.color();				
		}
	}
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
	//recursiveRegionGrowing();
	regionGrowing();
	//regionGrowingScan(4,15);		// #seeds, spacing ai lati (utile per spaziare i seed dai margini)
	
	return 0;
}
