#include <opencv2/ml/ml.hpp>
using namespace cv;

#include<iostream>
using namespace std;

//Debug
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

int csv2xml()
//int main()
{
	////¶ÁÈ¡CSVÊý¾Ý
	Ptr<ml::TrainData> train_data;
	train_data = ml::TrainData::loadFromCSV("./Resource/data/1.csv", 1);
	Mat m = train_data->getTrainSamples();

	imshow("CSV", m);
	cv::waitKey(1);

	return 0;
}
