
#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <math.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <stdlib.h>
#include <time.h>  

using namespace cv;
using namespace std;

Size previousSize;

class Cluster{
    public:
        vector<Point> pixels;
        Vec3b centroid;
        pair<Point, float> maxPoint;
        vector<float> distFromDataset;

        static Mat rawImage;
        static vector<Point> dataSet;

        Cluster(){
            centroid = Vec3b(0,0,0);
        }
        Cluster(Point point){
            centroid = getPixelVal(point);
        }
        ~Cluster(){}

        static void init(const char *fileName){
            rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
            if (rawImage.empty()){
                cout<<"--Error -- No image data"<<endl;
                exit(EXIT_FAILURE);
            }
            previousSize = rawImage.size();
	        cv::resize(rawImage, rawImage, cv::Size(100,100));
            
            for(int y=0; y<rawImage.rows; y++)
                for(int x=0; x<rawImage.cols; x++)
                    dataSet.push_back(Point(x,y));
        }


        static float euclideanDistance(Vec3b c1, Vec3b c2){
            return (pow(c1[0]-c2[0],2)+pow(c1[1]-c2[1],2)+pow(c1[2]-c2[2],2));
        }

        
        static Vec3b getPixelVal(Point point){
			return rawImage.at<Vec3b>(point);
		}

        static void pick2MaxPoints(Point& Y, Point& Z){
            int N = rawImage.rows;
            int M = rawImage.cols;
            float valMax = 0;
            for(int i=0; i<N*M; i++){
                for(int j=i+1; j<N*M; j++){
                    float distance = Cluster::euclideanDistance(Cluster::getPixelVal(dataSet.at(i)),Cluster::getPixelVal(dataSet.at(j)));
                    if(distance>valMax){
                        valMax = distance;
                        Y = dataSet.at(i);
                        Z = dataSet.at(j);
                    }
                }
            }
        }

        void insertData(int i){pixels.push_back(dataSet.at(i));}
        void clearData(){pixels.clear();}

        void calcCentroid(){
            if(pixels.size()!=0){
                float meanR, meanG, meanB;
                meanR = meanG = meanB = 0;
                vector<Point>::iterator p;
                for(p=pixels.begin(); p!=pixels.end(); p++){
                    Vec3b color = getPixelVal(*p);
                    meanB+=color[0];
                    meanG+=color[1];
                    meanR+=color[2];
                }
                meanB/=pixels.size();
                meanG/=pixels.size();
                meanR/=pixels.size();
                centroid = Vec3b(meanB,meanG,meanR);
            }
        }

        void calcMostDistantPoint(){
            float maxDist, distance;
            maxDist = 0;
            for(int i=0; i<pixels.size(); i++){
                distance = euclideanDistance(centroid,getPixelVal(pixels.at(i)));
                if(distance>maxDist){
                    maxDist = distance;
                    maxPoint = make_pair(pixels.at(i),maxDist);
                }
            }
        }

        void calcDistanceFromDataset(){
            distFromDataset.clear();
            for(int j=0; j<dataSet.size(); j++)
                distFromDataset.push_back(euclideanDistance(centroid,getPixelVal(dataSet.at(j))));
        }
        
        void color(Mat& inoutMatrix){
			vector<Point>::iterator it;
			for(it=pixels.begin(); it!=pixels.end(); it++){
				Point pixel = (*it);
				inoutMatrix.at<Vec3b>(pixel) = centroid;
			}
		}
};

Mat Cluster::rawImage;
vector<Point> Cluster::dataSet;


void MengHeeHang(){
    Point y,z;
    vector<Cluster> clusters, newClusters;
    vector<Cluster>::iterator c,c2;
    bool haveNewCluster;

    Cluster::pick2MaxPoints(y,z);
    clusters.push_back(Cluster(y));
    clusters.push_back(Cluster(z));

    do{
        haveNewCluster = false;

        for(c=clusters.begin(); c!=clusters.end(); c++){
            (*c).clearData();                   //rimuovi i pixel dal cluster (il centroide rimane)
            (*c).calcDistanceFromDataset();     //calcola la distanza di tutti i punti dal centroide (i punti sono appartenenti al dataset)
        }

        //per ogni elemento del dataset (pixel) trova il centroide del cluster a distanza minima
        for(int i=0; i<Cluster::dataSet.size(); i++){ 
            float minIndex;
            float minDistance = FLT_MAX;
            for(int k=0; k<clusters.size(); k++){
                if(clusters.at(k).distFromDataset.at(i) < minDistance){
                    minDistance = clusters.at(k).distFromDataset.at(i);
                    minIndex = k;
                }
            }
            //inserisci il pixel all'interno del cluster a distanza minima
            clusters.at(minIndex).insertData(i);
        }

        //per ogni cluster, ricalcola il centroide e il pixel X a distanza massima D
        for(c=clusters.begin(); c!=clusters.end();){
            if((*c).pixels.size() == 0){ //se dopo l'assegnazione dei pixel ai cluster, un cluster è vuoto, allora eliminalo
                c = clusters.erase(c);
            }else{
                (*c).calcCentroid();
                (*c).calcMostDistantPoint(); 
                ++c;
            }
        }
        
        float q, numCouples;
        q = numCouples = 0;
        for(int i=0; i<clusters.size(); i++){
            for(int j=i+1; j<clusters.size(); j++){
                q+=Cluster::euclideanDistance(clusters.at(i).centroid, clusters.at(j).centroid);
                numCouples++;
            }
        }
        q/=numCouples;

        //prendi il punto a massima distanza dai punti a più massima distanza dal proprio centroide
        Point mostDistantPoint;
        float d = 0;
        for(c=clusters.begin(); c!=clusters.end(); c++){
            float currDistance = (*c).maxPoint.second;
            if(currDistance>d){
                mostDistantPoint = (*c).maxPoint.first;
                d = currDistance;
            }
        }
        if(d>q/2){   
            clusters.push_back(Cluster(mostDistantPoint));
            haveNewCluster = true;
        }
    }while(haveNewCluster);
    
    Mat outputMatrix = Mat::zeros(Cluster::rawImage.size(),Cluster::rawImage.type());
    for(c=clusters.begin(); c!=clusters.end(); c++)
        (*c).color(outputMatrix);

	cv::resize(outputMatrix, outputMatrix, cv::Size(previousSize));
    imwrite("clusterized.png", outputMatrix);
}




int main(int argc, char** argv )
{
    if(argc!=2){
        cerr<<"-- Error -- Usage <program> <image.ext>"<<endl;
        exit(EXIT_FAILURE);
    }

    Cluster::init(argv[1]);
    MengHeeHang();

    return 0;
}