#pragma once

#include <opencv2/core.hpp>

#include "Function.h"


namespace meng
{
	class Net
	{
	public:
		//Integer vector specifying the number of neurons in each layer including the input and output layers.
		std::vector<int> layer_neuron_num;
		int output_interval = 10;

		std::string activation_funtion = "sigmoid";
		
		float learning_rate;
		float fine_tune_factor = 1.01;

		float accuracy = 0.;
		std::vector<double> loss_vec;

	protected:
		std::vector<cv::Mat> layer;
		std::vector<cv::Mat> weights;
		std::vector<cv::Mat> bias;
		std::vector<cv::Mat> delta_err;

		cv::Mat output_error;
		cv::Mat target;
		cv::Mat board;
		float loss;

	public:
		Net() {};
		~Net() {};

		//Initialize net:generate weights matrices,layer matrices and bias matrices
		// bias default all zero.
		void initNet(std::vector<int> layer_neuron_num_);

		//Initialize the weights matrices.
		void initWeights(int type = 0, double a = 0., double b = 0.1);

		//Initialize the bias matrices.
		void initBias(cv::Scalar& bias);

	};
}