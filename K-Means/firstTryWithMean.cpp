// RANDOM CENTROIDS

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

class Cluster{
    public:
        vector<Point> pixels;
        vector<float> centroid;
        vector<float> distFromDataset;

        static Mat rawImage;
        static vector<Point> dataSet;

        Cluster(){
            centroid={0,0,0};
            
        }
        ~Cluster(){}

        static void init(const char *fileName){
            rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
            if (rawImage.empty()){
                cout<<"--Error -- No image data"<<endl;
                exit(EXIT_FAILURE);
            }
            resize(rawImage,rawImage,Size(200,200));
            namedWindow("rawImage", WINDOW_AUTOSIZE);
            imshow("rawImage", rawImage);
            waitKey(0);
            for(int y=0; y<rawImage.rows; y++)
                for(int x=0; x<rawImage.cols; x++)
                    dataSet.push_back(Point(x,y));
        }
        
        static int getPixelVal(Point point, int channel){
			return rawImage.at<Vec3b>(point)[channel];
		}

        void insertData(int i){
            pixels.push_back(dataSet.at(i));
        }
        void clearData(){
            pixels.clear();
        }

        bool hasSimilarCentroidTo(vector<float> otherCentroid){
            float cWeight = centroid.at(0)+centroid.at(1)+centroid.at(2);
            float oWeight = otherCentroid.at(0)+otherCentroid.at(1)+otherCentroid.at(2);
            float threshold = 1;
           if(abs(cWeight-oWeight)<threshold)
                return true;
            return false;
        }

        void calcCentroid(){
            if(pixels.size()!=0){
                centroid = {0,0,0};
                vector<Point>::iterator p;
                for(p=pixels.begin(); p!=pixels.end(); p++){
                    centroid.at(0)+=getPixelVal(*p, 0);
                    centroid.at(1)+=getPixelVal(*p, 1);
                    centroid.at(2)+=getPixelVal(*p, 2);
                }
                centroid.at(0)/=pixels.size();
                centroid.at(1)/=pixels.size();
                centroid.at(2)/=pixels.size();
            }
        }

        void setRandomCentroid(){
            centroid.at(0) = rand()%255+1;
            centroid.at(1) = rand()%255+1;
            centroid.at(2) = rand()%255+1;
            // Point p(rand()%rawImage.rows, rand()%rawImage.cols);
            // centroid.at(0) = getPixelVal(p,0);
            // centroid.at(1) = getPixelVal(p,1);
            // centroid.at(2) = getPixelVal(p,2);
        }

        void calcDistanceFromDataset(){
            distFromDataset.clear();
            for(int j=0; j<dataSet.size(); j++){
                float distance = 0;
                distance+=pow(centroid.at(0)-Cluster::getPixelVal(dataSet.at(j),0),2);
                distance+=pow(centroid.at(1)-Cluster::getPixelVal(dataSet.at(j),1),2);
                distance+=pow(centroid.at(2)-Cluster::getPixelVal(dataSet.at(j),2),2);
                distFromDataset.push_back(sqrt(distance));
            }
        }
        
        void color(Mat& inoutMatrix){
			int R = centroid.at(2); //(rand()%255)+1;
			int G = centroid.at(1);//(rand()%255)+1;
			int B = centroid.at(0); //(rand()%255)+1;
			vector<Point>::iterator it;
			for(it=pixels.begin(); it!=pixels.end(); it++){
				Point pixel = (*it);
				inoutMatrix.at<Vec3b>(pixel) = Vec3b(B,G,R);
			}
		}

};

Mat Cluster::rawImage;
vector<Point> Cluster::dataSet;


void KMeans(int kClusters){
    vector<Cluster> clusters(kClusters);
    srand(time(NULL));
    vector<float> previousCentroid;
    vector<float> nextCentroid;
    vector<Cluster>::iterator c;
    bool converge;
    
    cout<<"-- RANDOM CENTROID --"<<endl;
    for(int i=0; i<clusters.size(); i++){
        clusters.at(i).setRandomCentroid();
        nextCentroid = clusters.at(i).centroid;
        cout<<"\tCentroid = "<<nextCentroid.at(0)<<","<<nextCentroid.at(1)<<","<<nextCentroid.at(2)<<endl;
    }
    cout<<endl<<"--- PROCESSING ---"<<endl;
    do{
        converge = true;
        for(int i=0; i<clusters.size(); i++){
            clusters.at(i).clearData();
            clusters.at(i).calcDistanceFromDataset();
        }
        for(int i=0; i<Cluster::dataSet.size(); i++){
            float minIndex;
            float minDistance = FLT_MAX;
            for(int k=0; k<clusters.size(); k++){
                if(clusters.at(k).distFromDataset.at(i) < minDistance){
                    minDistance = clusters.at(k).distFromDataset.at(i);
                    minIndex = k;
                }
            }
            clusters.at(minIndex).insertData(i);
        }
        
        for(c=clusters.begin(); c!=clusters.end(); c++){
            Cluster& curCluster = *c;
            previousCentroid = curCluster.centroid;
            curCluster.calcCentroid();
            nextCentroid = curCluster.centroid;
            if(!curCluster.hasSimilarCentroidTo(previousCentroid))
                converge = false;
        }
    }while(!converge);
    
    Mat outputMatrix = Mat::zeros(Cluster::rawImage.size(),Cluster::rawImage.type());
    for(c=clusters.begin(); c!=clusters.end(); c++)
        (*c).color(outputMatrix);
    imwrite("clusterized.png", outputMatrix);
}




int main(int argc, char** argv )
{
    if(argc!=3){
        cerr<<"-- Error -- Usage <program> <image.ext> <#clusters>"<<endl;
        exit(EXIT_FAILURE);
    }

    Cluster::init(argv[1]);
    KMeans(atoi(argv[2]));

    return 0;
}

/*
// SET CENTROIDS MANUALLY
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

class Cluster{
    public:
        vector<Point> pixels;
        vector<float> centroid;
        vector<float> distFromDataset;

        static Mat rawImage;
        static vector<Point> dataSet;

        Cluster(){
            centroid={0,0,0};
            
        }
        ~Cluster(){}

        static void init(const char *fileName){
            rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
            if (rawImage.empty()){
                cout<<"--Error -- No image data"<<endl;
                exit(EXIT_FAILURE);
            }

            namedWindow("rawImage", WINDOW_AUTOSIZE);
            imshow("rawImage", rawImage);
            waitKey(0);
            for(int y=0; y<rawImage.rows; y++)
                for(int x=0; x<rawImage.cols; x++)
                    dataSet.push_back(Point(x,y));
        }
        
        static int getPixelVal(Point point, int channel){
			return rawImage.at<Vec3b>(point)[channel];
		}

        void insertData(int i){
            pixels.push_back(dataSet.at(i));
        }
        void clearData(){
            pixels.clear();
        }

        bool hasSimilarCentroidTo(vector<float> otherCentroid){
            float cWeight = centroid.at(0)+centroid.at(1)+centroid.at(2);
            float oWeight = otherCentroid.at(0)+otherCentroid.at(1)+otherCentroid.at(2);
            float threshold = 10;
           if(abs(cWeight-oWeight)<threshold)
                return true;
            return false;
        }

        void calcCentroid(){
            if(pixels.size()!=0){
                vector<Point>::iterator p;
                for(p=pixels.begin(); p!=pixels.end(); p++){
                    centroid.at(0)+=getPixelVal(*p, 0);
                    centroid.at(1)+=getPixelVal(*p, 1);
                    centroid.at(2)+=getPixelVal(*p, 2);
                }
                centroid.at(0)/=pixels.size();
                centroid.at(1)/=pixels.size();
                centroid.at(2)/=pixels.size();
            }
        }

        void setCentroid(int R, int G, int B){
            centroid.at(0) = B;
            centroid.at(1) = G;
            centroid.at(2) = R;
        }

        void setRandomCentroid(){
            centroid.at(0) = rand()%255+1;
            centroid.at(1) = rand()%255+1;
            centroid.at(2) = rand()%255+1;
            // Point p(rand()%rawImage.rows, rand()%rawImage.cols);
            // centroid.at(0) = getPixelVal(p,0);
            // centroid.at(1) = getPixelVal(p,1);
            // centroid.at(2) = getPixelVal(p,2);
        }

        void calcDistanceFromDataset(){
            distFromDataset.clear();
            for(int j=0; j<dataSet.size(); j++){
                float distance = 0;
                distance+=pow(centroid.at(0)-Cluster::getPixelVal(dataSet.at(j),0),2);
                distance+=pow(centroid.at(1)-Cluster::getPixelVal(dataSet.at(j),1),2);
                distance+=pow(centroid.at(2)-Cluster::getPixelVal(dataSet.at(j),2),2);
                distFromDataset.push_back(sqrt(distance));
            }
        }
        
        void color(Mat& inoutMatrix){
			int R = (rand()%255)+1;
			int G = (rand()%255)+1;
			int B = (rand()%255)+1;
			vector<Point>::iterator it;
			for(it=pixels.begin(); it!=pixels.end(); it++){
				Point pixel = (*it);
				inoutMatrix.at<Vec3b>(pixel) = Vec3b(B,G,R);
			}
		}

};

Mat Cluster::rawImage;
vector<Point> Cluster::dataSet;


void KMeans(int kClusters){
    vector<Cluster> clusters(kClusters);
    srand(time(NULL));
    vector<float> previousCentroid;
    vector<float> nextCentroid;
    vector<Cluster>::iterator c;
    bool converge;
    for(int i=0; i<clusters.size(); i++){
        int R,G,B;
        cout<<"Insert cluster #"<<i+1<<"centroid: "<<endl;
        cout<<"R value: ";
        cin>>R;
        cout<<"G value: ";
        cin>>G;
        cout<<"B value: ";
        cin>>B;
        clusters.at(i).setCentroid(R,G,B);
        //clusters.at(i).setRandomCentroid();
    }
    cout<<endl<<"--- PROCESSING ---"<<endl;
    do{
        converge = true;
        for(int i=0; i<clusters.size(); i++){
            clusters.at(i).clearData();
            clusters.at(i).calcDistanceFromDataset();
        }
        for(int i=0; i<Cluster::dataSet.size(); i++){
            float minIndex;
            float minDistance = FLT_MAX;
            for(int k=0; k<clusters.size(); k++){
                if(clusters.at(k).distFromDataset.at(i) < minDistance){
                    minDistance = clusters.at(k).distFromDataset.at(i);
                    minIndex = k;
                }
            }
            clusters.at(minIndex).insertData(i);
        }
        
        for(c=clusters.begin(); c!=clusters.end(); c++){
            Cluster& curCluster = *c;
            previousCentroid = curCluster.centroid;
            curCluster.calcCentroid();
            nextCentroid = curCluster.centroid;
            if(!curCluster.hasSimilarCentroidTo(previousCentroid))
                converge = false;
        }
    }while(!converge);
    
    Mat outputMatrix = Mat::zeros(Cluster::rawImage.size(),Cluster::rawImage.type());
    for(c=clusters.begin(); c!=clusters.end(); c++)
        (*c).color(outputMatrix);
    imwrite("clusterized.png", outputMatrix);
}




int main(int argc, char** argv )
{
    if(argc!=3){
        cerr<<"-- Error -- Usage <program> <image.ext> <#clusters>"<<endl;
        exit(EXIT_FAILURE);
    }

    Cluster::init(argv[1]);
    KMeans(atoi(argv[2]));

    return 0;
}
*/

/*
// RANDOM CENTROIDS

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

class Cluster{
    public:
        vector<Point> pixels;
        vector<float> centroid;
        vector<float> distFromDataset;

        static Mat rawImage;
        static vector<Point> dataSet;

        Cluster(){
            centroid={0,0,0};
            
        }
        ~Cluster(){}

        static void init(const char *fileName){
            rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
            if (rawImage.empty()){
                cout<<"--Error -- No image data"<<endl;
                exit(EXIT_FAILURE);
            }

            namedWindow("rawImage", WINDOW_AUTOSIZE);
            imshow("rawImage", rawImage);
            waitKey(0);
            for(int y=0; y<rawImage.rows; y++)
                for(int x=0; x<rawImage.cols; x++)
                    dataSet.push_back(Point(x,y));
        }
        
        static int getPixelVal(Point point, int channel){
			return rawImage.at<Vec3b>(point)[channel];
		}

        void insertData(int i){
            pixels.push_back(dataSet.at(i));
        }
        void clearData(){
            pixels.clear();
        }

        bool hasSimilarCentroidTo(vector<float> otherCentroid){
            float cWeight = centroid.at(0)+centroid.at(1)+centroid.at(2);
            float oWeight = otherCentroid.at(0)+otherCentroid.at(1)+otherCentroid.at(2);
            float threshold = 10;
           if(abs(cWeight-oWeight)<threshold)
                return true;
            return false;
        }

        void calcCentroid(){
            if(pixels.size()!=0){
                vector<Point>::iterator p;
                for(p=pixels.begin(); p!=pixels.end(); p++){
                    centroid.at(0)+=getPixelVal(*p, 0);
                    centroid.at(1)+=getPixelVal(*p, 1);
                    centroid.at(2)+=getPixelVal(*p, 2);
                }
                centroid.at(0)/=pixels.size();
                centroid.at(1)/=pixels.size();
                centroid.at(2)/=pixels.size();
            }
        }

        void setRandomCentroid(){
            centroid.at(0) = rand()%255+1;
            centroid.at(1) = rand()%255+1;
            centroid.at(2) = rand()%255+1;
            // Point p(rand()%rawImage.rows, rand()%rawImage.cols);
            // centroid.at(0) = getPixelVal(p,0);
            // centroid.at(1) = getPixelVal(p,1);
            // centroid.at(2) = getPixelVal(p,2);
        }

        void calcDistanceFromDataset(){
            distFromDataset.clear();
            for(int j=0; j<dataSet.size(); j++){
                float distance = 0;
                distance+=pow(centroid.at(0)-Cluster::getPixelVal(dataSet.at(j),0),2);
                distance+=pow(centroid.at(1)-Cluster::getPixelVal(dataSet.at(j),1),2);
                distance+=pow(centroid.at(2)-Cluster::getPixelVal(dataSet.at(j),2),2);
                distFromDataset.push_back(sqrt(distance));
            }
        }
        
        void color(Mat& inoutMatrix){
			int R = (rand()%255)+1;
			int G = (rand()%255)+1;
			int B = (rand()%255)+1;
			vector<Point>::iterator it;
			for(it=pixels.begin(); it!=pixels.end(); it++){
				Point pixel = (*it);
				inoutMatrix.at<Vec3b>(pixel) = Vec3b(B,G,R);
			}
		}

};

Mat Cluster::rawImage;
vector<Point> Cluster::dataSet;


void KMeans(int kClusters){
    vector<Cluster> clusters(kClusters);
    srand(time(NULL));
    vector<float> previousCentroid;
    vector<float> nextCentroid;
    vector<Cluster>::iterator c;
    bool converge;
    for(int i=0; i<clusters.size(); i++)
        clusters.at(i).setRandomCentroid();
        
    do{
        converge = true;
        for(int i=0; i<clusters.size(); i++){
            clusters.at(i).clearData();
            clusters.at(i).calcDistanceFromDataset();
        }
        for(int i=0; i<Cluster::dataSet.size(); i++){
            float minIndex;
            float minDistance = FLT_MAX;
            for(int k=0; k<clusters.size(); k++){
                if(clusters.at(k).distFromDataset.at(i) < minDistance){
                    minDistance = clusters.at(k).distFromDataset.at(i);
                    minIndex = k;
                }
            }
            clusters.at(minIndex).insertData(i);
        }
        
        for(c=clusters.begin(); c!=clusters.end(); c++){
            Cluster& curCluster = *c;
            previousCentroid = curCluster.centroid;
            curCluster.calcCentroid();
            nextCentroid = curCluster.centroid;
            if(!curCluster.hasSimilarCentroidTo(previousCentroid))
                converge = false;
        }
    }while(!converge);
    
    Mat outputMatrix = Mat::zeros(Cluster::rawImage.size(),Cluster::rawImage.type());
    for(c=clusters.begin(); c!=clusters.end(); c++)
        (*c).color(outputMatrix);
    imwrite("clusterized.png", outputMatrix);
}




int main(int argc, char** argv )
{
    if(argc!=3){
        cerr<<"-- Error -- Usage <program> <image.ext> <#clusters>"<<endl;
        exit(EXIT_FAILURE);
    }

    Cluster::init(argv[1]);
    KMeans(atoi(argv[2]));

    return 0;
}
*/

/*
//CENTROIDI RANDOM CON QUALCHE COMMENTO IN PIU'


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

class Cluster{
    public:
        vector<Point> pixels;
        vector<float> centroid;
        vector<float> distFromDataset;

        static Mat rawImage;
        static vector<Point> dataSet;

        Cluster(){
            centroid={0,0,0};
            
        }
        ~Cluster(){}

        static void init(const char *fileName){
            rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
            if (rawImage.empty()){
                cout<<"--Error -- No image data"<<endl;
                exit(EXIT_FAILURE);
            }

            namedWindow("rawImage", WINDOW_AUTOSIZE);
            imshow("rawImage", rawImage);
            waitKey(0);
            for(int y=0; y<rawImage.rows; y++)
                for(int x=0; x<rawImage.cols; x++)
                    dataSet.push_back(Point(x,y));
        }
        
        static int getPixelVal(Point point, int channel){
			return rawImage.at<Vec3b>(point)[channel];
		}

        void insertData(int i){
            pixels.push_back(dataSet.at(i));
        }
        void clearData(){
            pixels.clear();
        }

        bool hasSimilarCentroidTo(vector<float> otherCentroid){
            float cWeight = centroid.at(0)+centroid.at(1)+centroid.at(2);
            float oWeight = otherCentroid.at(0)+otherCentroid.at(1)+otherCentroid.at(2);
            float threshold = 10;
            cout<<endl<<"\tabs(cWeight-oWeight)="<<abs(cWeight-oWeight)<<endl;
            if(abs(cWeight-oWeight)<threshold)
                return true;
            return false;
        }

        void calcCentroid(){
            if(pixels.size()!=0){
                vector<Point>::iterator p;
                for(p=pixels.begin(); p!=pixels.end(); p++){
                    centroid.at(0)+=getPixelVal(*p, 0);
                    centroid.at(1)+=getPixelVal(*p, 1);
                    centroid.at(2)+=getPixelVal(*p, 2);
                }
                centroid.at(0)/=pixels.size();
                centroid.at(1)/=pixels.size();
                centroid.at(2)/=pixels.size();
            }
        }

        void setRandomCentroid(){
            centroid.at(0) = rand()%255+1;
            centroid.at(1) = rand()%255+1;
            centroid.at(2) = rand()%255+1;
            // Point p(rand()%rawImage.rows, rand()%rawImage.cols);
            // centroid.at(0) = getPixelVal(p,0);
            // centroid.at(1) = getPixelVal(p,1);
            // centroid.at(2) = getPixelVal(p,2);
        }

        void calcDistanceFromDataset(){
            distFromDataset.clear();
            for(int j=0; j<dataSet.size(); j++){
                float distance = 0;
                distance+=pow(centroid.at(0)-Cluster::getPixelVal(dataSet.at(j),0),2);
                distance+=pow(centroid.at(1)-Cluster::getPixelVal(dataSet.at(j),1),2);
                distance+=pow(centroid.at(2)-Cluster::getPixelVal(dataSet.at(j),2),2);
                distFromDataset.push_back(sqrt(distance));
            }
        }
        
        void color(Mat& inoutMatrix){
			int R = (rand()%255)+1;
			int G = (rand()%255)+1;
			int B = (rand()%255)+1;
			vector<Point>::iterator it;
			for(it=pixels.begin(); it!=pixels.end(); it++){
				Point pixel = (*it);
				inoutMatrix.at<Vec3b>(pixel) = Vec3b(B,G,R);
			}
		}

};

Mat Cluster::rawImage;
vector<Point> Cluster::dataSet;


void KMeans(int kClusters){
    vector<Cluster> clusters(kClusters);
    srand(time(NULL));

    vector<float> previousCentroid;
    vector<float> nextCentroid;
    vector<Cluster>::iterator c;


    cout<<"-- RANDOM CENTROID --"<<endl;
    for(int i=0; i<clusters.size(); i++){
        clusters.at(i).setRandomCentroid();
        nextCentroid = clusters.at(i).centroid;
        cout<<"\tCentroid = "<<nextCentroid.at(0)<<","<<nextCentroid.at(1)<<","<<nextCentroid.at(2)<<endl;
    }
    cout<<endl;
    
    bool converge;
    int iter = 0;
    do{
        converge = true;
        cout<<endl<<"--- Iteration # "<<++iter<<" ---"<<endl;
        for(int i=0; i<clusters.size(); i++){
            clusters.at(i).clearData();
            clusters.at(i).calcDistanceFromDataset();
        }
        
        for(int i=0; i<Cluster::dataSet.size(); i++){
            float minIndex;
            float minDistance = FLT_MAX;
            for(int k=0; k<clusters.size(); k++){
                //clusters.at(i).calcDistanceFromDataset();
                if(clusters.at(k).distFromDataset.at(i) < minDistance){
                    minDistance = clusters.at(k).distFromDataset.at(i);
                    minIndex = k;
                }
            }
            clusters.at(minIndex).insertData(i);
        }
        
        for(c=clusters.begin(); c!=clusters.end(); c++){
            Cluster& curCluster = *c;
            previousCentroid = curCluster.centroid;
            cout<<"--- BEFORE ---"<<endl;
            cout<<"\tCentroid = "<<previousCentroid.at(0)<<","<<previousCentroid.at(1)<<","<<previousCentroid.at(2)<<endl;
            
            curCluster.calcCentroid();
            nextCentroid = curCluster.centroid;
            cout<<"--- AFTER ---"<<endl;
            cout<<"\tCentroid = "<<nextCentroid.at(0)<<","<<nextCentroid.at(1)<<","<<nextCentroid.at(2);
            cout<<endl;

            if(!curCluster.hasSimilarCentroidTo(previousCentroid))
                converge = false;
        }
    }while(!converge);
    
    Mat outputMatrix = Mat::zeros(Cluster::rawImage.size(),Cluster::rawImage.type());
    string clustName = "OutputCluster";
    int i = 0;
    for(c=clusters.begin(); c!=clusters.end(); c++){
        (*c).color(outputMatrix);
        
        i++;
    }
    namedWindow(clustName+to_string(i), WINDOW_AUTOSIZE);
        imshow(clustName+to_string(i), outputMatrix);
        waitKey(0);

    imwrite("clusterized.png", outputMatrix);
}




int main(int argc, char** argv )
{
    if(argc!=3){
        cerr<<"-- Error -- Usage <program> <image.ext> <#clusters>"<<endl;
        exit(EXIT_FAILURE);
    }

    Cluster::init(argv[1]);
    KMeans(atoi(argv[2]));
    // imshow("original", image);
    //cvtColor(image,image,COLOR_BGR2HSV);
    //resize(image,image,Size(MIN(image.rows,image.cols),MIN(image.rows,image.cols)),0,0,INTER_LINEAR);
    //outputHSV = Mat(image.size(),CV_8UC3);
    //cvtColor(outputHSV,outputHSV,COLOR_BGR2HSV);
    //GaussianBlur(image,image, Size(5,5), 3, 3, 4);


    return 0;
}
*/

/*
// I CENTROIDI ME LI CALCOLO INSERENDO ALL'INIZIO TUTTI I PIXEL NEI CLUSTER

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

class Cluster{
    public:
        vector<Point> pixels;
        vector<float> centroid;
        vector<float> distFromDataset;

        static Mat rawImage;
        static vector<Point> dataSet;

        Cluster(){
            centroid={0,0,0};
            
        }
        ~Cluster(){}

        static void init(const char *fileName){
            rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
            if (rawImage.empty()){
                cout<<"--Error -- No image data"<<endl;
                exit(EXIT_FAILURE);
            }

            namedWindow("rawImage", WINDOW_AUTOSIZE);
            imshow("rawImage", rawImage);
            waitKey(0);
            for(int y=0; y<rawImage.rows; y++)
                for(int x=0; x<rawImage.cols; x++)
                    dataSet.push_back(Point(x,y));
        }
        
        static int getPixelVal(Point point, int channel){
			return rawImage.at<Vec3b>(point)[channel];
		}

        void insertData(int i){
            pixels.push_back(dataSet.at(i));
        }

        bool hasSimilarCentroidTo(vector<float> otherCentroid){
            float cWeight = centroid.at(0)+centroid.at(1)+centroid.at(2);
            float oWeight = otherCentroid.at(0)+otherCentroid.at(1)+otherCentroid.at(2);
            float threshold = 0.5;
            cout<<endl<<"\tabs(cWeight-oWeight)="<<abs(cWeight-oWeight)<<endl;
            if(abs(cWeight-oWeight)<threshold)
                return true;
            
            // if( abs(centroid.at(0)-otherCentroid.at(0))<threshold
            //     && abs(centroid.at(1)-otherCentroid.at(1))<threshold
            //     && abs(centroid.at(2)-otherCentroid.at(2))<threshold
            // )
            //     return true;
            
            return false;
        }

        void calcCentroid(){
            if(pixels.size()!=0){
                vector<Point>::iterator p;
                for(p=pixels.begin(); p!=pixels.end(); p++){
                    centroid.at(0)+=getPixelVal(*p, 0);
                    centroid.at(1)+=getPixelVal(*p, 1);
                    centroid.at(2)+=getPixelVal(*p, 2);
                }
                centroid.at(0)/=pixels.size();
                centroid.at(1)/=pixels.size();
                centroid.at(2)/=pixels.size();
            }
        }

        void setRandomCentroid(){
            Point p(rand()%rawImage.rows, rand()%rawImage.cols);
            centroid.at(0) = getPixelVal(p,0);
            centroid.at(1) = getPixelVal(p,1);
            centroid.at(2) = getPixelVal(p,2);
        }

        void calcDistanceFromDataset(){
            for(int j=0; j<dataSet.size(); j++){
                float distance = 0;
                distance+=pow(centroid.at(0)-Cluster::getPixelVal(dataSet.at(j),0),2);
                distance+=pow(centroid.at(1)-Cluster::getPixelVal(dataSet.at(j),1),2);
                distance+=pow(centroid.at(2)-Cluster::getPixelVal(dataSet.at(j),2),2);
                distFromDataset.push_back(sqrt(distance));
            }
        }

        void erasePixels(){
            pixels.clear();
        }
        
        void color(Mat& inoutMatrix){
			int R = (rand()%255)+1;
			int G = (rand()%255)+1;
			int B = (rand()%255)+1;
			vector<Point>::iterator it;
			for(it=pixels.begin(); it!=pixels.end(); it++){
				Point pixel = (*it);
				inoutMatrix.at<Vec3b>(pixel) = Vec3b(B,G,R);
			}
		}

};

Mat Cluster::rawImage;
vector<Point> Cluster::dataSet;


void KMeans(int kClusters){
    vector<Cluster> clusters(kClusters);
    srand(time(NULL));

    vector<float> previousCentroid;
    vector<float> nextCentroid;
    vector<Cluster>::iterator c;


    for(int i=0; i<kClusters; i++)
        clusters.at(i) = Cluster();

    for(int i=0; i<Cluster::dataSet.size(); i++){
        int randomIndex = rand()%kClusters;
        clusters.at(randomIndex).insertData(i);
    }

    for(int i=0; i<kClusters; i++){
        clusters.at(i).calcCentroid();
        nextCentroid = clusters.at(i).centroid;
        cout<<"\tCentroid = "<<nextCentroid.at(0)<<","<<nextCentroid.at(1)<<","<<nextCentroid.at(2)<<endl;
    }
    
    

    bool converge;
    int iter = 0;
    do{
            //riassegna i pixel con media più vicina
        for(int i=0; i<kClusters; i++){
            clusters.at(i).erasePixels();
            clusters.at(i).calcDistanceFromDataset();
        }
        converge = true;
        cout<<endl<<"--- Iteration # "<<++iter<<" ---"<<endl;
        for(int i=0; i<Cluster::dataSet.size(); i++){
            float minIndex;
            float minDistance = FLT_MAX;
            for(int k=0; k<clusters.size(); k++){
                if(clusters.at(k).distFromDataset.at(i) < minDistance){
                    minDistance = clusters.at(k).distFromDataset.at(i);
                    minIndex = k;
                }
            }
            clusters.at(minIndex).insertData(i);
        }
        
        for(c=clusters.begin(); c!=clusters.end(); c++){
            Cluster& curCluster = *c;
            previousCentroid = curCluster.centroid;
            cout<<"--- BEFORE ---"<<endl;
            cout<<"\tCentroid = "<<previousCentroid.at(0)<<","<<previousCentroid.at(1)<<","<<previousCentroid.at(2)<<endl;
            
            curCluster.calcCentroid();
            nextCentroid = curCluster.centroid;
            cout<<"--- AFTER ---"<<endl;
            cout<<"\tCentroid = "<<nextCentroid.at(0)<<","<<nextCentroid.at(1)<<","<<nextCentroid.at(2);
            cout<<endl;

            if(!curCluster.hasSimilarCentroidTo(previousCentroid))
                converge = false;
        }
    }while(!converge);
    Mat outputMatrix = Mat::zeros(Cluster::rawImage.size(),Cluster::rawImage.type());
    int i = 0;
    for(c=clusters.begin(); c!=clusters.end(); c++){
        (*c).color(outputMatrix);
        cout<<"SIZE: "<<(*c).pixels.size()<<endl;
    }
    imwrite("clusterized.png", outputMatrix);
    
}




int main(int argc, char** argv )
{
    if(argc!=3){
        cerr<<"-- Error -- Usage <program> <image.ext> <#clusters>"<<endl;
        exit(EXIT_FAILURE);
    }

    Cluster::init(argv[1]);
    KMeans(atoi(argv[2]));
    return 0;
}
*/