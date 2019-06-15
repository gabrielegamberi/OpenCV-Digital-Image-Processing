#include <stdio.h>
#include <iostream>
#include <string>
#include <math.h>
#include <time.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

class Cluster{
	public:
		static Size backupSize;
		static Mat sourceMatrix;
		static vector<Point> dataSet;
		vector<Point> points;
		pair<Point, float> maxPoint;
		Vec3b centroid;
		vector<float> distancesFromCentroid;
		
		Cluster(){centroid = Vec3b(0,0,0);}
		Cluster(Point point){centroid = getPixelValue(point);}

		inline static Vec3b getPixelValue(Point point){return sourceMatrix.at<Vec3b>(point);}

		static float euclideanDistance(Vec3b color1, Vec3b color2){
			return pow(color1[0]-color2[0],2)+pow(color1[1]-color2[1],2)+pow(color1[2]-color2[2],2);
		}

		static void pick2MaxPoints(Point& Y, Point& Z){
			int N = sourceMatrix.rows;
			int M = sourceMatrix.cols;
			float valMax = 0;
			for(int i=0; i<dataSet.size(); i++){
				for(int j=i+1; j<dataSet.size(); j++){
					float distance = euclideanDistance(getPixelValue(dataSet.at(i)), getPixelValue(dataSet.at(j)));
					if(distance>valMax){
						valMax = distance;
						Y = dataSet.at(i);
						Z = dataSet.at(j);
					}
				}
			}
			
		}

		static void restorePictureSize(){
			resize(Cluster::sourceMatrix,Cluster::sourceMatrix,backupSize);
			dataSet.clear();
			fillDataset();
		}
		
		static void fillDataset(){
			for(int y=0; y<sourceMatrix.rows; y++)
                for(int x=0; x<sourceMatrix.cols; x++)
                    dataSet.push_back(Point(x,y));
		}

		static void init(String fileName){
			sourceMatrix = imread(fileName, IMREAD_COLOR);
			if(sourceMatrix.empty()){
				cerr<<"--- ERROR --- matrix is empty!"<<endl;
				exit(EXIT_FAILURE);
			}
			backupSize = sourceMatrix.size();
			resize(Cluster::sourceMatrix,Cluster::sourceMatrix,Size(100,100));
			
			fillDataset();

			namedWindow("sourceMatrix", WINDOW_AUTOSIZE);
			imshow("sourceMatrix", sourceMatrix);
		}

		void clearPoints(){points.clear();}

		void calcDistancesFromOwnCentroid(){
			distancesFromCentroid.clear();
			for(int i=0; i<dataSet.size(); i++)
				distancesFromCentroid.push_back(euclideanDistance(centroid, getPixelValue(dataSet.at(i))));
		}

		void insertPoint(Point newPoint){
			points.push_back(newPoint);
		}
		inline int getSize(){points.size();}

		void calcMostDistantPoint(){
			float maxDistance, distance;
			maxDistance = 0;
			for(int i=0; i<points.size(); i++){
				distance = euclideanDistance(centroid, getPixelValue(points.at(i)));
				if(distance>maxDistance){
					maxDistance = distance;
					maxPoint = make_pair(points.at(i), maxDistance);
				}
			}
		}

		void calcCentroid(){
			if(points.size()>0){
				float meanR, meanG, meanB;
				meanR = meanG = meanB = 0;
				for(int i=0; i<points.size(); i++){
					Vec3b color = getPixelValue(points.at(i));
					meanB+=color[0];
					meanG+=color[1];
					meanR+=color[2];
				}
				meanB/=points.size();
				meanG/=points.size();
				meanR/=points.size();
				centroid = Vec3b(meanB, meanG, meanR);
			}
		}

		void color(Mat &outputMatrix){
			for(int i=0; i<points.size(); i++)
				outputMatrix.at<Vec3b>(points.at(i)) = centroid;
		}
};

Mat Cluster::sourceMatrix;
Size Cluster::backupSize;
vector<Point> Cluster::dataSet;


void MengHeeHang(){
	vector<Cluster> clusters;
	vector<Cluster>::iterator c;
	Point Y,Z;

	Cluster::pick2MaxPoints(Y,Z);
	clusters.push_back(Cluster(Y));
	clusters.push_back(Cluster(Z));
	cout<<"--- Max point calculated (chromatically speaking)---"<<endl;
	cout<<"Y ("<<Y.y<<","<<Y.x<<")\t|\tZ ("<<Z.y<<","<<Z.x<<")"<<endl;

	Cluster::restorePictureSize();
	cout<<"--- Image resized back to "<<Cluster::backupSize<<" ---"<<endl;
	
	cout<<"--- Processing ---"<<endl;
	int iteration = 0;
	bool haveNewCluster;
	do{
		cout<<"- Iteration #"<<(++iteration)<<endl;
		haveNewCluster = false;
		//per ogni cluster: resetta i punti e calcola la distanza di ogni pixel del dataset dal rispettivo centroide 
		for(c=clusters.begin(); c!=clusters.end(); c++){
			(*c).clearPoints();
			(*c).calcDistancesFromOwnCentroid();
		}

		//determina il cluster a distanza minima
		for(int i=0; i<Cluster::dataSet.size(); i++){
			int iMinDistance;
			float minDistance = FLT_MAX;
			for(int k=0; k<clusters.size(); k++){
				Cluster& current = clusters.at(k);
				if(current.distancesFromCentroid.at(i)<minDistance){
					minDistance = current.distancesFromCentroid.at(i);
					iMinDistance = k;
				}
			}
			clusters.at(iMinDistance).insertPoint(Cluster::dataSet.at(i));
		}

		//per ogni cluster ricalcola il centroide
		for(c=clusters.begin(); c!=clusters.end();){
			Cluster& cluster = (*c);
			if(cluster.getSize()==0){
				c = clusters.erase(c);
			}else{
				cluster.calcCentroid();
				cluster.calcMostDistantPoint();
				++c;
			}
		}

		//calcolati la distanza media "q" tra i centroidi (somma le distanza tra ogni coppia di centroidi e fai la media)
		float q, numCouples;
		q = numCouples = 0;
		for(int i=0; i<clusters.size(); i++){
			for(int j=i+1; j<clusters.size(); j++){
				q+=Cluster::euclideanDistance(clusters.at(i).centroid, clusters.at(j).centroid);
				numCouples++;
			}
		}
		q/=numCouples;
		
		//prendi il punto che si trova a massima distanza dal proprio centroide (X: MASSIMO DEI MASSIMI)
		Point X;
		float d = 0;
		for(c=clusters.begin(); c!=clusters.end(); c++){
			float currentDistance = (*c).maxPoint.second; //prendi la distanza del punto massimo
			if(currentDistance > d){
				d = currentDistance;
				X = (*c).maxPoint.first; 			//prendi il punto massimo
			}
		}	

		//se viene rispettata questa condizione, allora X sarà messo in un nuovo cluster
		if(d>q/2){
			clusters.push_back(Cluster(X));
			haveNewCluster = true;
		}
	}while(haveNewCluster);

	Mat outputMatrix = Mat::zeros(Cluster::sourceMatrix.size(), Cluster::sourceMatrix.type());
	for(c=clusters.begin(); c!=clusters.end(); c++)
		(*c).color(outputMatrix);

	imwrite("clusterized.png", outputMatrix);
}

int main(int argc, char** argv ){
	if(argc!=2){
		cerr<<"--- ERROR --- required <executable> <filename.ext>"<<endl;
		exit(EXIT_FAILURE);
	}	
	Cluster::init(argv[1]);
	MengHeeHang();
	//Mat sourceMatrix = imread(argv[1], IMREAD_COLOR);
	//namedWindow("sourceMatrix", WINDOW_AUTOSIZE);
	//imshow("sourceMatrix", sourceMatrix);
	waitKey(0);
	return 0;
}