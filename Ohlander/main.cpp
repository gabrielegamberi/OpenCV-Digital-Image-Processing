#include <stdio.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <opencv2/opencv.hpp>

#define DELTA(size) round(size/5)
#define LOWER_DELTA_THRESHOLD 10

using namespace std;
using namespace cv;

Mat rawImage, blurredImage;
vector<Mat> finalClusters;

class Sample{
    public:
        int occurrence;      //y dell'istogramma
        int chromaticity;   //x dell'istogramma
        
        Sample(){}
        Sample(int occurrence, int chromaticity){
            this->occurrence = occurrence;
            this->chromaticity = chromaticity;
        }
        bool operator<(const Sample &nextSample)const{
			return (occurrence>nextSample.occurrence);
		}
};

class OhlanderCore{
    public:
		vector<float> histograms[3];
        vector<Sample> orderedChannels[3];
        int deltas[3];
        int valley;
        int selectedChannel;

        OhlanderCore(Mat mask){
            initHist(mask);
            selectedChannel = -1;
            valley = INT_MAX;
            calcDeltas();
        }

        void initHist(Mat mask){
			//inizializzo a zero gli istogrammi
			for(int channel=0; channel<3; channel++)
				histograms[channel] = vector<float>(256, 0);

			for(int row=0; row<blurredImage.rows; row++)
				for(int col=0; col<blurredImage.cols; col++)
					if(mask.at<uchar>(row,col)==1)
						for(int channel=0; channel<3; channel++)
							histograms[channel].at(blurredImage.at<Vec3b>(row,col)[channel])++;	
			
            //popola i canali (BGR) con dei Sample (memorizzo in esso la sua cromaticità e la sua occorrenza)  
            for(int c=0; c<3; c++)
                for(int i=0; i<histograms[c].size(); i++)
                    orderedChannels[c].push_back(Sample((int)histograms[c].at(i), i));				
			
			//ordinali in senso decrescente (override dell'operatore nella classe Sample)
            for(int c=0; c<3; c++)
                sort(orderedChannels[c].begin(), orderedChannels[c].end());		
		}


        void calcDeltas(){
            int leftIndex,rightIndex;
            for(int c=0; c<3; c++){
                for(int l=0; l<histograms[c].size(); l++){
                    if(histograms[c].at(l)!=0){
                        leftIndex = l;
                        break;
                    }
                }
                for(int r=histograms[c].size()-1; r>=0; r--){
                    if(histograms[c].at(r)!=0){
                        rightIndex = r;
                        break;
                    }
                }
                deltas[c] = DELTA((rightIndex-leftIndex));
            }
        }

        void tryClustering(){
            pair<Sample, Sample> hills[3];
            Sample max, secondMax;
            //determina i picchi (hills) di ogni canale
            for(int c=0; c<3; c++){
                max = orderedChannels[c].at(0);
                secondMax = max;
                for(int i=1; i<histograms[c].size(); i++){
                    //se il picco successivo dista DELTA dal massimo, allora consideralo come secondo massimo
                    if(abs(max.chromaticity-orderedChannels[c].at(i).chromaticity)>deltas[c]){
                        secondMax = orderedChannels[c].at(i);
                        break;
                    }
                }
                hills[c] = make_pair(max, secondMax);
            }
            //determina quale canale considerare per il clustering (istogramma con picco massimo ed una valle)
            int maxOccurrence = 0;
            for(int c=0; c<3; c++){
                //se l'istogramma al canale C ha 2 picchi
                if(hills[c].first.chromaticity != hills[c].second.chromaticity){
                    if(hills[c].first.occurrence > maxOccurrence){
                        maxOccurrence = hills[c].first.occurrence;
                        selectedChannel = c;
                    }
                }
            }
            if(isClusterizable())
                valley = round(abs(hills[selectedChannel].first.chromaticity+hills[selectedChannel].second.chromaticity)/2);
        }

        inline bool isClusterizable(){return selectedChannel>=0 && deltas[selectedChannel]>=LOWER_DELTA_THRESHOLD;} 
};


void createMasks(const Mat currMask, Mat& firstMask, Mat& secondMask, int channel, int chromaThreshold){
    firstMask = Mat::zeros(blurredImage.size(), CV_8UC1);
    secondMask = Mat::zeros(blurredImage.size(), CV_8UC1);
    for(int i=0; i<blurredImage.rows; i++){
        for(int j=0; j<blurredImage.cols; j++){
            //Se la posizione corrente è attiva (mascherata)
            if(currMask.at<uchar>(i,j)==1){
                if(blurredImage.at<Vec3b>(i,j)[channel] < chromaThreshold){
                    firstMask.at<uchar>(i,j) = 1;
                    secondMask.at<uchar>(i,j) = 0;
                } else{
                    firstMask.at<uchar>(i,j) = 0;
                    secondMask.at<uchar>(i,j) = 1;
                }
            }
        }
    }
}

void Ohlander(const Mat mask){
    OhlanderCore maskedRegion(mask);
    maskedRegion.tryClustering();
    if(maskedRegion.isClusterizable()){
        Mat firstMask, secondMask;
        createMasks(mask, firstMask, secondMask, maskedRegion.selectedChannel, maskedRegion.valley);
        Ohlander(firstMask);
        Ohlander(secondMask);
    }else{
        finalClusters.push_back(mask);
    }
}



int main(int argc, char** argv ){
    if(argc!=2){
        cerr<<"--- ERROR --- Required: <fileName> <image.ext>"<<endl;
        exit(EXIT_FAILURE);
    }
    const char* fileName = argv[1];
    rawImage = imread(fileName, IMREAD_COLOR);//immagine presa in input
    if (rawImage.empty()){
        cerr<<"--- ERROR --- No data found"<<endl;
        exit(EXIT_FAILURE);
    }
    
    blurredImage = rawImage.clone();
    GaussianBlur(rawImage, blurredImage, Size(5,5) , 8);    //più è alta la varianza, più è forte lo smoothing
    Ohlander(Mat::ones(blurredImage.size(), CV_8UC1));

    cout <<"Number of clusters: "<<finalClusters.size() << endl;
    srand(time(NULL));
    Mat tempMatrix = Mat::zeros(blurredImage.size(), blurredImage.type());
    for(int i=0; i<finalClusters.size(); i++){
        Vec3b color(rand()%255,rand()%255,rand()%255);
        for(int row=0; row<tempMatrix.rows; row++){
            for(int col=0; col<tempMatrix.cols; col++){
                if(finalClusters.at(i).at<uchar>(row,col)!=0){
                    tempMatrix.at<Vec3b>(row,col) = color;
                }
            }   
        }
    }
    namedWindow("FinalImage", WINDOW_AUTOSIZE );
    imshow("FinalImage", tempMatrix );
    waitKey(0);
    return 0;
}