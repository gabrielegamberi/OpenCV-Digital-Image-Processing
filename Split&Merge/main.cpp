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
#define AREA_THRESHOLD 10*10
#define VARIANCE_THRESHOLD 50

Mat image;
Mat outputBW;
Mat outputBGR;

class Region
{
    public:
        float mean;
        float var;
        Rect square;
        set<Region*> adj;

    Region(Point first, Point second)
    {
        square = Rect(first,second);  // (200,200) - (250, 250)
        calcMean();
        calcVar();
    }
    static Point getMidPoint(Region& region, int side);
    static void mapAdjacents();
    void calcMean();
    void calcVar();
    bool isUniform();
   
};
vector<Region> regions;

void Region::calcMean()
{
    mean = 0;
    int rowStart = square.y;
    int rowEnd = square.y + square.height;
    int colStart = square.x;
    int colEnd = square.x + square.width;
    for(int i=rowStart; i<=rowEnd; i++)
    {
        for(int j=colStart; j<=colEnd; j++)
        {
            mean += image.at<uchar>(i,j);
        }
    }
    mean = mean/((square.height+1)*(square.width+1));
}

void Region::calcVar()
{
    var = 0;
    int rowStart = square.y;
    int rowEnd = square.y + square.height;
    int colStart = square.x;
    int colEnd = square.x + square.width;
    for(int i=rowStart; i<=rowEnd; i++)
    {
        for(int j=colStart; j<=colEnd; j++)
        {
            var+=pow((image.at<uchar>(i,j)-mean),2);
        }
    }
    var = var/((square.height+1)*(square.width+1));
}

bool Region::isUniform()
{
    if(var<VARIANCE_THRESHOLD)
        return true;
    return false;
}

Point Region::getMidPoint(Region& region, int side)
{
    int midX, midY;
    if(side==0)//TOP
    {
            midX = (region.square.x+(region.square.x+region.square.width))/2;
            midY = region.square.y-2;
    }
    else if(side==1)//RIGHT
    {
            midX = region.square.x+2;
            midY = (region.square.y+(region.square.y+region.square.height))/2;
    }
    else if(side==2)//BOTTOM
    {
            midX = (region.square.x+(region.square.x+region.square.width))/2;
            midY = region.square.y+2;
    }
    else if(side==3)//LEFT
    {
            midX = region.square.x-2;
            midY = (region.square.y+(region.square.y+region.square.height))/2;
    }
    return Point(midX,midY);
}

void Region::mapAdjacents()
{
    //sort(regions.begin(), regions.end(), [](Region& l, Region& r) { return l.square.height < r.square.height;});
    for(int k=0; k<regions.size(); k++)
    {
        vector<Point> midPoints;
        Region& current = regions.at(k);
        for(int side=0; side<4; side++)
        {
            midPoints.push_back(getMidPoint(current,side));
        }
        for(int l=0; l<regions.size(); l++) //per ogni altra regione
        {
            Region& neighbour = regions.at(l);
            if(k!=l && (current.square.height<=neighbour.square.height))//SE NON TI STAI CONFRONTANDO CON TE STESSO E SONO PIU' PICCOLO DI TE
            {
                for(int side=0; side<4; side++)
                {
                    if(neighbour.square.contains(midPoints.at(side)))
                    {
                        current.adj.insert(&neighbour);
                        neighbour.adj.insert(&current);
                        break;
                    }
                }
            }
        }
    }
}
//____________________________________________________________________________________

void split(int rowStart, int rowEnd, int colStart, int colEnd)
{
    Region current(Point(rowStart,colStart),Point(rowEnd,colEnd));
    //CASO BASE - LIMITE DIMENSIONI REGIONE
    if((rowEnd-rowStart+1)*(colEnd-colStart+1) <= AREA_THRESHOLD)
    {
        regions.push_back(current);
        return;
    }
    //CASO BASE - UNIFORMITA'
    if(current.isUniform())
    {
        regions.push_back(current);
        return;
    }
    else
    {
        split(rowStart, ((rowStart+rowEnd)/2), colStart, ((colStart+colEnd)/2));//1
        split(rowStart, ((rowStart+rowEnd)/2), ((colStart+colEnd)/2)+1, colEnd);//2
        split(((rowStart+rowEnd)/2)+1, rowEnd, colStart, ((colStart+colEnd)/2));//3
        split(((rowStart+rowEnd)/2)+1, rowEnd, ((colStart+colEnd)/2)+1, colEnd);//4
    }
}

void merge()
{
    for(int i=0; i<regions.size(); i++)
    {
        Region& current = regions.at(i);
        for(set<Region*>::iterator it=current.adj.begin(); it!=current.adj.end(); it++)
        {
            Region& neighbour = *(*it);
            if(abs(current.mean-neighbour.mean)<30)
                neighbour.mean = current.mean;
        }
    }
}

void print()
{
    srand(time(NULL));
    unsigned char colorBW;
    Vec3b colorBGR;
    for(int k=0; k<regions.size(); k++)
    {
        colorBGR = Vec3b(rand()%256,rand()%256,rand()%256);
        colorBW = regions.at(k).mean;
        for(int i=regions.at(k).square.y; i<regions.at(k).square.y+regions.at(k).square.height; i++)
            for(int j=regions.at(k).square.x; j<regions.at(k).square.x+regions.at(k).square.width; j++)
                outputBW.at<uchar>(i,j) = colorBW;
                //outputBGR.at<Vec3b>(i,j) = colorBGR;
    }
    // imwrite("output.png", outputBGR);
}

void splitAndMerge()
{
    int N = image.rows;
    int M = image.cols;
    split(0, N-1, 0, M-1);
    print();
    imshow("before", outputBW);
    Region::mapAdjacents();
    merge();
    print();
    imshow("after", outputBW);
    waitKey(0);
}

int main(int argc, char** argv )
{
    const char* fileName = "house.jpg";
    image = imread(fileName, IMREAD_GRAYSCALE);//immagine presa in input
    if (!image.data)
    {
        printf("No image data \n");
        return -1;
    }
    outputBW = Mat::zeros(image.size(), CV_8UC1);
    outputBGR = Mat(image.size(),CV_8UC3);
    GaussianBlur(image,image, Size(5,5), 3, 3, 4);
    splitAndMerge();
    return 0;
}