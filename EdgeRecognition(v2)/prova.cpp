#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

#define SIZE 3
#define PAD floor(SIZE/2)

/* RICORDA
Mat kernel(LEN,LEN,CV_8U,Scalar(1));
Mat kernel = Mat::ones(LEN,LEN,CV_8U);
Mat mask = (Mat_<char>(LEN,LEN)<<1,1,1
                                 1,1,1
                                 1,1,1
);
*/
Mat vertPrewitt = (Mat_<char>(SIZE,SIZE)<<-1,0,1,-1,0,1,-1,0,1);
Mat horPrewitt = (Mat_<char>(SIZE,SIZE)<<-1,-1,-1,0,0,0,1,1,1);


void equalize(Mat& inputMatrix){
	int L = 256;
	int N = inputMatrix.rows;
	int M = inputMatrix.cols;
	vector<int> pixelCount(L,0);
	for(int i=0; i<N; i++)
		for(int j=0; j<M; j++)
			pixelCount.at(inputMatrix.at<uchar>(i,j))++;
	vector<int> equalizedPixel(L,0);
	for(int k=0; k<L; k++){
		int sumPixel = 0;
		for(int j=0; j<k; j++)
			sumPixel+=pixelCount.at(j);
		equalizedPixel.at(k) = ((float)(L-1)/(M*N))*sumPixel;
	}
	for(int row=0; row<N; row++)
		for(int col=0; col<M; col++)
			inputMatrix.at<uchar>(row,col) = equalizedPixel.at(inputMatrix.at<uchar>(row,col));
}


int getMin(int a, int b){
	return ((a<b)? (a):(b));
}
int getMax(int a, int b){
	return ((a>b)? (a):(b));
}



void erode_dilate_matrix(Mat& inoutPaddedMatrix, String selectedFilter){
	int initValue;
	int (*filter)(int,int);
	int N = inoutPaddedMatrix.rows;
	int M = inoutPaddedMatrix.cols;
	Mat tempMatrix = inoutPaddedMatrix.clone();
	if(selectedFilter == "erode"){
		initValue = UCHAR_MAX;
		filter = getMin;
	}else{
		initValue = 0;
		filter = getMax;
	}
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			int returnValue = initValue;
			for(int i=-PAD; i<=PAD; i++){
				for(int j=-PAD; j<=PAD; j++){
					int currentPixel = tempMatrix.at<uchar>(row+i,col+j);
					returnValue = filter(currentPixel, returnValue);
				}
			}
			inoutPaddedMatrix.at<uchar>(row,col) = returnValue;
		}
	}

}

int getMedian(int value[]){
    int n = SIZE*SIZE;
    int iMin;
    for(int i=0; i<n-1; i++){
        iMin = i;
        for(int j=i+1; j<n; j++)
            if(value[j]<value[iMin])
                iMin = j;
		int temp = value[iMin];
		value[iMin] = value[i];
		value[i] = temp;
	}
    return value[n/2];
}

void medianFilter(const Mat paddedMatrix, Mat& outputMatrix){
	int maskedValue[SIZE*SIZE];
    for(int y=+PAD; y<paddedMatrix.rows-PAD; y++){
        for(int x=+PAD; x<paddedMatrix.cols-PAD; x++){
			int mask_iter = 0;
			for(int row=-PAD; row<=PAD; row++)
				for(int col=-PAD; col<=PAD; col++)
					maskedValue[mask_iter++] = paddedMatrix.at<uchar>(row+y, col+x);
            outputMatrix.at<uchar>(y,x) = getMedian(maskedValue);    //assegno al pixel corrente la media dei pixel nell'intorno
        }
    }
}


void prewittEdges(const Mat paddedMatrix, Mat& outputMatrix){
	int N = paddedMatrix.rows;
	int M = paddedMatrix.cols;
	int vertAccum, horAccum;
	int threshold = 100;
	
	outputMatrix = paddedMatrix.clone();
	
	medianFilter(paddedMatrix, outputMatrix); //attenuo il rumore
	
	Mat tempMatrix = outputMatrix.clone();
	equalize(tempMatrix);					//equalizzo l'immagine
	
	namedWindow("medianFilter", WINDOW_AUTOSIZE);
    imshow("medianFilter", outputMatrix); 
	namedWindow("equalized", WINDOW_AUTOSIZE);
    imshow("equalized", tempMatrix);   

	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			vertAccum = horAccum = 0;
			for(int i=-PAD; i<=PAD; i++){
				for(int j=-PAD; j<=PAD; j++){
					vertAccum+=(int)(tempMatrix.at<uchar>(row+i,col+j)*vertPrewitt.at<char>(i+PAD,j+PAD));
					horAccum+=(int)(tempMatrix.at<uchar>(row+i,col+j)*horPrewitt.at<char>(i+PAD,j+PAD));
				}
			}
			horAccum = (horAccum>threshold)? 255:0;
			vertAccum = (vertAccum>threshold)? 255:0;
			outputMatrix.at<uchar>(row,col) = max(horAccum, vertAccum);
		}
	}
	erode_dilate_matrix(outputMatrix, "dilate");
	erode_dilate_matrix(outputMatrix, "erode");				//con questa combo cerco di unire gli edge
}


void buildPaddedMatrix(const Mat inputMatrix, Mat& paddedMatrix){
	int corner = 0;
	int row, col;
	paddedMatrix = Mat::zeros(inputMatrix.rows+2*PAD, inputMatrix.cols+2*PAD, inputMatrix.type());
	for(row=PAD; row<paddedMatrix.rows-PAD; row++){
		for(col=PAD; col<paddedMatrix.cols-PAD; col++){
			paddedMatrix.at<uchar>(row,col) = inputMatrix.at<uchar>(row-PAD, col-PAD);
		}
	}
	while(corner<4){
		switch(corner){
			case 0://copia colonna di sinistra
				for(row=PAD; row<paddedMatrix.rows-PAD; row++)
					for(col=0; col<PAD; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(row,PAD);
				break;
			case 1://colonna di destra
				for(row=PAD; row<paddedMatrix.rows-PAD; row++)
					for(col=paddedMatrix.cols-PAD; col<paddedMatrix.cols; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(row,paddedMatrix.cols-PAD-1);
				break;
			case 2://riga superiore
				for(row=0; row<PAD; row++)
					for(col=0; col<paddedMatrix.cols; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(PAD,col);
				break;
			case 3://riga inferiore
				for(row=paddedMatrix.rows-PAD; row<paddedMatrix.rows; row++)
					for(col=0; col<paddedMatrix.cols; col++)
						paddedMatrix.at<uchar>(row,col) = paddedMatrix.at<uchar>(paddedMatrix.rows-PAD-1, col);
				break;
		}

		corner++;
	}


}

int main(int argc, char** argv )
{
   	const char* fileName = (argc>=2)? argv[1]:"./lena.jpg";
	Mat inputMatrix = imread(fileName, IMREAD_GRAYSCALE);
	if(inputMatrix.empty()){
		cerr<<"-- Matrix is empty --"<<endl;
		exit(EXIT_FAILURE);
	}

	Mat paddedMatrix, noPaddedMatrix, outputMatrix;
	buildPaddedMatrix(inputMatrix, paddedMatrix);
	prewittEdges(paddedMatrix, outputMatrix);

	outputMatrix(Rect(Point(PAD, outputMatrix.cols-PAD),Point(outputMatrix.rows-PAD, PAD))).copyTo(noPaddedMatrix);

	namedWindow("input", WINDOW_AUTOSIZE);
	imshow("input", inputMatrix);
	namedWindow("output", WINDOW_AUTOSIZE);
	imshow("output", noPaddedMatrix);
	waitKey(0);
	return 0;
}
