#include "Function.h"

#include <iostream>

namespace meng
{

	cv::Mat sigmoid(cv::Mat &x)
	{
		cv::Mat exp_x, fx;

		cv::exp(-x, exp_x);
		fx = 1.0 / (1.0 + exp_x);

		return fx;
	}

	void calcLoss(cv::Mat &output, cv::Mat &target, cv::Mat &output_error, float &loss)
	{
		if (target.empty())
		{
			std::cout << "Can't find the target cv::Matrix" << std::endl;
			return;
		}

		output_error = target - output;
		cv::Mat err_sqrare;
		pow(output_error, 2, err_sqrare);
		cv::Scalar err_sqr_sum = sum(err_sqrare);
		loss = err_sqr_sum[0] / (float)(output.rows);
	}

	cv::Mat derivativeFuntion(cv::Mat& fx, std::string func_type)
	{
		cv::Mat dx;

		if (func_type == "sigmoid")
		{
			dx = sigmoid(fx).mul(1 - sigmoid(fx));
		}

		return dx;
	}

}