#pragma once
#include<opencv2/core.hpp>

namespace meng
{
	//Sigmoid function
	cv::Mat sigmoid(cv::Mat &x);

	//Objective function
	void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss);

	//Derivative function
	cv::Mat derivativeFuntion(cv::Mat& fx, std::string func_type);
}