//Esempio di utilizzo: ./prova fruits.jpg 10
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
		vector<Point> points;					//vettore che contiene i punti del cluster
		vector<float> centroid; 				//media delle 3 componenti: (B,G,R)
		vector<float> distancesFromCentroid; 	//distanza di ogni pixel dal proprio centroide

		static Mat sourceMatrix;				//immagine di partenza
		static vector<Point> dataSet;			//vettore che contiene TUTTI i pixel dell'immagine

		Cluster(){centroid = {0,0,0};}
		~Cluster(){}

		//rimuovi tutti i punti dal cluster (pulizia)
		void clearPoints(){
			points.clear();
		}

		//aggiungi al cluster l'i-simo pixel/punto del dataset
		void insertPointFromData(int i){
			points.push_back(dataSet.at(i));
		}

		//calcola il nuovo centroide sulla base delle 3 componenti (B,G,R)
		void calcCentroid(){
			if(points.size()!=0){
				centroid = {0,0,0};
				for(int i=0; i<points.size(); i++){
					centroid.at(0)+=getPixelValAt(points.at(i), 0);
					centroid.at(1)+=getPixelValAt(points.at(i), 1);
					centroid.at(2)+=getPixelValAt(points.at(i), 2);
				}
				centroid.at(0)/=points.size();
				centroid.at(1)/=points.size();
				centroid.at(2)/=points.size();
			}
		}

		//calcola le distanze di tutti i pixel dell'immagine dal centroide
		void calcDistancesFromOwnCentroid(){
			distancesFromCentroid.clear();
			for(int i=0; i<dataSet.size(); i++){
				float distance = 0;
				distance+=pow(centroid.at(0)-getPixelValAt(dataSet.at(i),0),2);
				distance+=pow(centroid.at(1)-getPixelValAt(dataSet.at(i),1),2);
				distance+=pow(centroid.at(2)-getPixelValAt(dataSet.at(i),2),2);
				distancesFromCentroid.push_back(sqrt(distance));
			}
		}

		//inizializzatore
		static void init(String fileName){
			sourceMatrix = imread(fileName, IMREAD_COLOR);
			if(sourceMatrix.empty()){
				cerr<<"--- ERROR --- Source matrix is empty"<<endl;
				exit(EXIT_FAILURE);
			}
			resize(sourceMatrix, sourceMatrix, Size(150,150));
			namedWindow("sourceMatrix", WINDOW_AUTOSIZE);
			imshow("sourceMatrix", sourceMatrix);

			//riempi il dataset con i pixel dell'immagine
			for(int row=0; row<sourceMatrix.rows; row++)
				for(int col=0; col<sourceMatrix.cols; col++)
					dataSet.push_back(Point(col,row));
		}

		//prendi il valore del PUNTO p nel canale CHANNEL
		inline static int getPixelValAt(Point p, int channel){
			return sourceMatrix.at<Vec3b>(p)[channel];
		}

		//confronta il centroide con un altro centroide (utile per stabilire la loro similarità)
		bool hasSimilarCentroidTo(vector<float> otherCentroid){
			float currentWeight = centroid.at(0)+centroid.at(1)+centroid.at(2);
			float otherWeight = otherCentroid.at(0)+otherCentroid.at(1)+otherCentroid.at(2);
			int threshold = 2;
			if(abs(currentWeight-otherWeight)<threshold)
				return true;							//sono simili
			return false;								//sono molto diversi
		}

		//imposta un centroide random
		void setRandomCentroid(){
			centroid.at(0) = rand()%256;
			centroid.at(1) = rand()%256;
			centroid.at(2) = rand()%256;
		}

		//colora la matrice "inoutMatrix" con i valori RGB del centroide
		void color(Mat &inoutMatrix){
			int R = centroid.at(2);
			int G = centroid.at(1);
			int B = centroid.at(0);
			for(int i=0; i<points.size(); i++)
				inoutMatrix.at<Vec3b>(points.at(i)) = Vec3b(B,G,R);
		}

};

Mat Cluster::sourceMatrix;
vector<Point> Cluster::dataSet;


void KMeans(int kClusters){
	srand(time(NULL));

	vector<float> previousCentroid;
	vector<Cluster>::iterator c;
	vector<Cluster> clusters(kClusters);

	//Imposta i centroidi (random) dei K cluster
	cout<<"--- RANDOM CENTROID ---"<<endl;
	for(int i=0; i<clusters.size(); i++)
		clusters.at(i).setRandomCentroid();

	cout<<"--- PROCESSING ---"<<endl;
	bool converge;
	do{
		converge = true;
		//ripulisci i cluster
		for(c=clusters.begin(); c!=clusters.end(); c++){
			Cluster &current = *c;
			current.clearPoints(); 					//rimuovi i suoi punti
			current.calcDistancesFromOwnCentroid(); //calcola le distanze di ciascun pixel dell'immagine dal rispettivo centroide
		}

		//per ogni pixel del dataset, determina il cluster più vicino
		for(int k=0; k<Cluster::dataSet.size(); k++){
			int iMinCluster; 					//indice del cluster a distanza minima
			float minDistance = FLT_MAX;
			for(int i=0; i<clusters.size(); i++){ //controlla la distanza del pixel K dal cluster I-simo
				float actualDistance = clusters.at(i).distancesFromCentroid.at(k);
				if(actualDistance<minDistance){
					minDistance = actualDistance;
					iMinCluster = i;
				}
			}
			clusters.at(iMinCluster).insertPointFromData(k); //inserisci il K-simo punto del cluster a distanza minima
		}
		
		//per ogni cluster controlla la convergenza
		for(c=clusters.begin(); c!=clusters.end(); c++){
			Cluster &current = *c;
			previousCentroid = current.centroid;
			current.calcCentroid();					//ricalcola il centroide
			if(!current.hasSimilarCentroidTo(previousCentroid)) //se i due centroidi si differenziano di molto
				converge = false;					//non siamo arrivati ancora a convergenza
		}
	}while(!converge);	//continua il loop se NON sei arrivato a CONVERGENZA

	Mat outputMatrix = Mat::zeros(Cluster::sourceMatrix.size(), Cluster::sourceMatrix.type());
	for(int i=0; i<clusters.size(); i++)
		clusters.at(i).color(outputMatrix);
	
    imwrite("clusterized.png", outputMatrix);
}


int main(int argc, char** argv ){
	if(argc!=3){
		cerr<<"--- ERROR --- required <executable> <filename.ext> <numberOfClusters>"<<endl;
		exit(EXIT_FAILURE);
	}	
	Cluster::init(argv[1]);
	KMeans(atoi(argv[2]));
	waitKey(0);
	return 0;
}