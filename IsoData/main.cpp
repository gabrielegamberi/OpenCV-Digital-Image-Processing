#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define K 5                 //indentifica il numero di cluster iniziali
#define VAR_THRESHOLD 500   //se aumenta, diminuiscono il numero di cluster (aumentiamo l'eterogeneità)
#define SIZE_THRESHOLD 10   //se aumenta, ammettiamo più cluster nel processo di fusione
#define SUB_CLUSTERS 3      //nella fase di splitting dividiamo il cluster in 3

class Pixel{
    public:
        Point coords;
        Vec3b color;

    Pixel(){}

    Pixel(Point p, Vec3b color){
        this->coords = p;
        this->color = color;
    }
};


class Cluster
{
    public:
        Vec3b mean;
        vector<Pixel> pixels;

    Cluster(Pixel pixel):mean(pixel.color){}
    
    inline void addPixel(Pixel pixel){pixels.push_back(pixel);}

    void calculateMean()
    {
        Vec3f mean;
        for(auto pixel: pixels)
        {
            mean[0]+=pixel.color[0];
            mean[1]+=pixel.color[1];
            mean[2]+=pixel.color[2];
        }
        mean[0] = mean[0]/pixels.size();
        mean[1] = mean[1]/pixels.size();
        mean[2] = mean[2]/pixels.size();
        
        this->mean = mean;
    }

    //Calcolo e restituisco la varianza per ogni cluster
    float getVar(){
        Vec3f variance;
        for(int i=0; i<pixels.size(); i++){
            //accumulo la varianza di tutti i pixel per ogni canale
            variance[0] += pow( pixels.at(i).color[0]-this->mean[0], 2);
            variance[1] += pow( pixels.at(i).color[1]-this->mean[1], 2);
            variance[2] += pow( pixels.at(i).color[2]-this->mean[2], 2);
        }
        variance[0] = variance[0] / pixels.size();
        variance[1] = variance[1] / pixels.size();
        variance[2] = variance[2] / pixels.size();
        return ((variance[0]+variance[1]+variance[2])/3);
    }

    inline int getSize(){return this->pixels.size();}

    inline void color(Mat &image){
        for(auto pixel: this->pixels){
            Point &coords = pixel.coords;
            image.at<Vec3b>(coords) = this->mean;
        }
    }
    //calcolo la distanza euclidea tra due punti nei tre canali RGB
    inline static float euclideanDistance(Vec3b p1, Vec3b p2){
        return cv::norm(p1,p2);
    }


};

Mat image;
vector<Pixel> pixels;
vector<Cluster> clusters;


void populateClusters(vector<Cluster>& inoutClusters, const vector<Pixel> dataSet){   
    float distance, minDistance;
    int iMinDist;
    for(int i=0; i<inoutClusters.size(); i++)
        inoutClusters.at(i).pixels.clear();
    //calcola la distanza euclidea di ogni pixel dai centroidi di cluster e assegna il pixel al cluster più vicino
    for(int i=0; i<dataSet.size(); i++){
        minDistance = FLT_MAX;
        for(int j=0; j<inoutClusters.size(); j++){
            distance = Cluster::euclideanDistance(inoutClusters.at(j).mean, dataSet.at(i).color);
            if(distance < minDistance){
                iMinDist = j;
                minDistance = distance;
            }
        }
        //aggiungi il pixel al cluster più vicino
        inoutClusters.at(iMinDist).addPixel(dataSet.at(i));
    }
    for(int i=0; i<inoutClusters.size(); i++)
        inoutClusters.at(i).calculateMean();
}

void init(){
    int N = image.rows;
    int M = image.cols;
    //inserisci tutti pixel in un vettore
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            pixels.push_back(Pixel(Point(j,i), image.at<Vec3b>(i, j)));
    //Inserisci K pixel equidistribuiti lungo il vettore N*M
    for(int k=0; k<K; k++)
        clusters.push_back(Cluster(Pixel(pixels.at(k*((N-1)*(M-1)/K)))));
    //Fase di riempimento dei clusters
    populateClusters(clusters, pixels);
}


void splitCluster(vector<int>& indCluster){
    vector<Pixel> tempPixels;
    vector<Cluster> tempClusters;
    vector<Pixel> dataSet;
    sort(indCluster.begin(), indCluster.end(), [](int l, int r ) {return l>r;} );
    //prendo gli indici di ogni cluster da dividere
    for(int i=0; i<indCluster.size(); i++){
        int index = indCluster.at(i);
        tempPixels = clusters.at(index).pixels;
        dataSet.insert(dataSet.end(), tempPixels.begin(), tempPixels.end());
        clusters.erase(clusters.begin()+index);
        //scorro il vettore per prendere dei pixel da dare ai nuovi cluster
        for(int i=0; i<SUB_CLUSTERS; i++){
            //scelgo tre pixel di partenza per tre nuovi cluster prendendoli dal cluster che dobbiamo dividere
            Pixel& centroids = tempPixels.at(i*tempPixels.size()/SUB_CLUSTERS);
            //salvo i nuovi cluster con i pixel scelti
            tempClusters.push_back(centroids);
        }
    }
    //popoliamo i clusters divisi
    populateClusters(tempClusters, dataSet);
    clusters.insert(clusters.end(), tempClusters.begin(), tempClusters.end());
    indCluster.clear();
}

void mergeCluster(vector<int>& indCluster){
    int n = indCluster.size();
    //ordiniamo gli indici in senso decrescente per evitare cancellazioni fuori range dal vettore clusters
    sort(indCluster.begin(), indCluster.end(), [](int l, int r ) {return l>r;} );
    //se i cluster sono dispari (es. n=5) allora fai il merge dei primi n-1 cluster (n-1=4), mentre l'n-simo lo si lascia nei cluster 
    if(n%2 != 0)
        n--;
    for(int i=0; i<n; i+=2){
        vector<Pixel> &curr = clusters.at(indCluster.at(i)).pixels;
        vector<Pixel> &next = clusters.at(indCluster.at(i+1)).pixels;
        next.insert(next.end(), curr.begin(), curr.end());
        clusters.erase(clusters.begin()+indCluster.at(i));
    }
    indCluster.clear();
}


void isodata(){
    //1. inizializza k clusters di partenza
    init();
    bool converge;
    vector<int> indCluster;
    do{
        converge = false;
        for(int i=0; i<clusters.size(); i++){
            float var = clusters.at(i).getVar();
            if(var > VAR_THRESHOLD){
                converge = true;
                //accumulo in un vettore tutti i cluster con varianza troppo grande
                indCluster.push_back(i);
            }
        }
        //2. Divido i cluster con varianza eccessiva in cluster più piccoli
        splitCluster(indCluster);
        
        //3. Prendi i cluster più piccoli ed uniscili
        for(int i=0; i<clusters.size(); i++){
            if(clusters.at(i).getSize() < SIZE_THRESHOLD){
                converge = true;
                indCluster.push_back(i);
            }
        }
        mergeCluster(indCluster);
        for(int i=0; i<clusters.size(); i++)
            clusters.at(i).calculateMean();   
    }while(converge);
    
    for(auto c: clusters)
        c.color(image);

    namedWindow("result", WINDOW_AUTOSIZE);
    imshow("result", image);
    waitKey(0);
    imwrite("clusterized.png", image);
}

int main(int argc, char** argv ){
    if(argc!=2){
        cerr<<"--- ERROR --- Required: <fileName> <image.ext>"<<endl;
        exit(EXIT_FAILURE);
    }
    const char* fileName = argv[1];
    image = imread(fileName, IMREAD_COLOR);//immagine presa in input
    if (image.empty()){
        cerr<<"--- ERROR --- No data found"<<endl;
        exit(EXIT_FAILURE);
    }
    isodata();
    return 0;
}