#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <iostream>
#include <list>
/*HARRIS CON L'UTILIZZO DEGLI EIGEN VAL (FUNZIONANTE)*/
using namespace cv;
using namespace std;

#define SIZE 5
#define PAD floor(SIZE/2)
#define SOBEL_SIZE 3
#define SOBEL_PAD floor(SOBEL_SIZE/2)

Mat sourceMatrix, grayMatrix;
Mat covMatrix, harrisEigenVal;

const char* harrisWindow = "My Harris corner detector";

Mat kernel(SIZE,SIZE,CV_32FC1,Scalar(0));

Mat horSobel = (Mat_<char>(SOBEL_SIZE,SOBEL_SIZE)<<-1,-2,-1,
									   				 0, 0, 0,
									   				 1, 2, 1
);
Mat vertSobel = (Mat_<char>(SOBEL_SIZE,SOBEL_SIZE)<<-1, 0, 1,
									    			-2, 0, 2,
									    			-1, 0, 1
);

class Pixel{
	private:
		int row;
		int col;
		float eigenVal;
	public:
		Pixel(int newRow, int newCol, int newEigenVal): row(newRow), col(newCol), eigenVal(newEigenVal){}
		bool operator<(const Pixel &anotherPixel)const{
			return (anotherPixel.eigenVal<eigenVal);
		}
		inline float getEigenVal(){return this->eigenVal;}
		inline int getRow(){return this->row;}
		inline int getCol(){return this->col;}
};

void displayHistogram(const Mat& image){
    int N = image.rows;
    int M = image.cols;
	int L = 256;
    vector<int> histogram(L,0);
    for(int i=0; i<N; i++)
        for(int j=0; j<M; j++)
            histogram.at(image.at<uchar>(i,j))++;
    //FIND THE MAX VALUE OF THE BINS
    int max = histogram.at(0);
    for(int i = 1; i < L; i++)
        if(max < histogram.at(i))
            max = histogram.at(i);
    int hist_width = 1024;
    int hist_height = 400;
    int bin_widht = round((float)hist_width/L);
    Mat histogramImage(hist_height+24, hist_width+48, CV_8UC1, Scalar(255, 255, 255));
    for(int i=0; i<L; i++)
        histogram.at(i) = ((double)histogram.at(i)/max)*hist_height;
    //DRAW THE LINE
    for(int i = 0; i < L; i++)
      //line(Mat& img, Point pt1, Point pt2, const Scalar& color, int thickness=1, int lineType=8, int shift=0)
        line(histogramImage, Point(bin_widht*(i)+24, hist_height), Point(bin_widht*(i)+24, hist_height - histogram.at(i)),Scalar(0,0,0), 2, 8, 0);
    imshow("histogram", histogramImage);
    
}

void buildGaussianFilter(){
	float variance = 4;
	int rowShift, colShift, B;
	float minTerm;
	for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			kernel.at<float>(i,j) = exp(-(pow(abs(i-PAD),2)+pow(abs(j-PAD),2))/(2*variance));
	minTerm = kernel.at<float>(kernel.rows-1,kernel.cols-1);

	for(int i=0; i<kernel.rows; i++){
		for(int j=0; j<kernel.cols; j++){
			B = round(kernel.at<float>(i,j)/minTerm);
			kernel.at<float>(i,j) = B;	
		}
	}
}

void gaussianFilter(Mat& inoutPaddedMatrix){
	int denom=0;
	Mat temp = inoutPaddedMatrix.clone();

	for(int i=0; i<kernel.rows; i++)
		for(int j=0; j<kernel.cols; j++)
			denom+=kernel.at<uchar>(i,j);

	int maskedValue[SIZE*SIZE];
	for(int row=PAD; row<temp.rows-PAD; row++){
		for(int col=PAD; col<temp.cols-PAD; col++){
			int mask_index = 0;
			for(int i=-PAD; i<=PAD; i++)
				for(int j=-PAD; j<=PAD; j++)
					maskedValue[mask_index++] = temp.at<uchar>(row+i,col+j)*kernel.at<float>(i+PAD, j+PAD);
			int sum=0;
			for(int i=0; i<SIZE*SIZE; i++)
				sum+=maskedValue[i];
			sum=floor(sum/denom);
			inoutPaddedMatrix.at<uchar>(row,col) = sum;
		}
	}
}

void sobel(const Mat& inputMatrix, Mat& sobelGx, Mat& sobelGy){
	int yGrad, xGrad;
	int N = inputMatrix.rows;
	int M = inputMatrix.cols;
	sobelGx = Mat(inputMatrix.size(), CV_8U);	
	sobelGy = Mat(inputMatrix.size(), CV_8U);
	
	for(int row=PAD; row<N-PAD; row++){
		for(int col=PAD; col<M-PAD; col++){
			yGrad = xGrad = 0;
			for(int i=-SOBEL_PAD; i<=SOBEL_PAD; i++){
				for(int j=-SOBEL_PAD; j<=SOBEL_PAD; j++){
					yGrad+=(int)(inputMatrix.at<uchar>(row+i,col+j)*vertSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
					xGrad+=(int)(inputMatrix.at<uchar>(row+i,col+j)*horSobel.at<char>(i+SOBEL_PAD,j+SOBEL_PAD));
				}
			}
			sobelGx.at<uchar>(row,col) = saturate_cast<uchar>(abs(xGrad));
			sobelGy.at<uchar>(row,col) = saturate_cast<uchar>(abs(yGrad));
		}
	}
	namedWindow("X",WINDOW_AUTOSIZE);
	imshow("X", sobelGx);
	namedWindow("Y",WINDOW_AUTOSIZE);
	imshow("Y", sobelGy);
	
}

float getEigenValue(int row, int col, const Mat& gx, const Mat& gy, int windowSize){
	int pad = floor(windowSize/2);
	float k = 0.04f;
	Mat covarianceMat = Mat::zeros(2,2,CV_32F);
	for(int i=-pad; i<=pad; i++){
		for(int j=-pad; j<=pad;j ++){
			covarianceMat.at<float>(0,0)+=(float)pow(gx.at<uchar>(row+i,col+j),2);
			covarianceMat.at<float>(0,1)+=(float)gx.at<uchar>(row+i,col+j)*gy.at<uchar>(row+i,col+j);
			covarianceMat.at<float>(1,0)+=(float)gx.at<uchar>(row+i,col+j)*gy.at<uchar>(row+i,col+j);
			covarianceMat.at<float>(1,1)+=(float)pow(gy.at<uchar>(row+i,col+j),2);
			
		}
	}
	Mat eValue;
	cv::eigen(covarianceMat,eValue);
	float l1,l2, minLambda;
	l1 = eValue.at<float>(0,0);
	l2 = eValue.at<float>(1,0);
	minLambda = ((l1<l2)? (l1):(l2));
	if(minLambda>20)
		return minLambda;
	else
		return 0;	
}
void suppressCorners(list<Pixel> responses, Mat &inoutMatrix){
	responses.sort();
	list<Pixel>::iterator it,it2;
	int neighborhood = 15;
	int pad = floor(neighborhood/2);
	float limit = sqrt(pow(pad,2)+pow(pad,2));
	for(it=responses.begin();it!=responses.end();){
		Pixel currentCorner = *it;
		it2 = it;
		advance(it2,1);
		for(; it2!=responses.end();){
			Pixel neighbor = *it2;
			float pixelDistance = sqrt(pow(neighbor.getRow()-currentCorner.getRow(),2)+pow(neighbor.getCol()-currentCorner.getCol(),2));
			if(pixelDistance<=limit){
				it2 = responses.erase(it2);
			}else{
				++it2;
			}
		}
		++it;
	}
	for(it=responses.begin(); it!=responses.end(); it++)
		circle(inoutMatrix, Point(it->getCol(),it->getRow()), pad, Scalar(150), 1, 4);
}

void detectHarrisCorners(){
	Mat paddedMatrix = Mat::zeros(grayMatrix.rows+2*PAD, grayMatrix.cols+2*PAD, grayMatrix.type());
	Mat sobelGx, sobelGy;

	for(int i=PAD; i<paddedMatrix.rows-PAD; i++)
		for(int j=PAD; j<paddedMatrix.cols-PAD; j++)
			paddedMatrix.at<uchar>(i,j) = grayMatrix.at<uchar>(i-PAD, j-PAD);

	buildGaussianFilter();
	sobel(paddedMatrix, sobelGx, sobelGy);
	
    gaussianFilter(sobelGx);
    gaussianFilter(sobelGy);

	Mat beforeMatrix = paddedMatrix.clone();
	harrisEigenVal = Mat(paddedMatrix.size(), CV_32FC1);
	
	list<Pixel> responses;
	int neighborhood = 5;
	for(int row=PAD; row<paddedMatrix.rows-PAD; row++){
		for(int col=PAD; col<paddedMatrix.cols-PAD; col++){
			harrisEigenVal.at<float>(row,col) = getEigenValue(row,col,sobelGx,sobelGy,neighborhood);
			Pixel pixel(row,col,harrisEigenVal.at<float>(row,col));
			if(pixel.getEigenVal()!=0){
				circle(beforeMatrix, Point(col,row), 5, Scalar(150), 1, 4);
				responses.push_front(pixel);
			}
		}
	}

	namedWindow("before",WINDOW_AUTOSIZE);
	imshow("before", beforeMatrix);
	
	suppressCorners(responses, paddedMatrix);
	
	namedWindow("after",WINDOW_AUTOSIZE);
	imshow("after", paddedMatrix);
}

int main( int argc, char** argv ){
    const char* fileName = (argc>=2)? argv[1]:"./scacchi.jpg";
	sourceMatrix = imread(fileName, IMREAD_COLOR);
    
	if (sourceMatrix.empty()){
        cout<<"--- EMPTY IMAGE ---"<<endl;
        exit(EXIT_FAILURE);
    }

    cvtColor(sourceMatrix, grayMatrix, COLOR_BGR2GRAY );
    detectHarrisCorners();

    waitKey(0);
    return 0;
}