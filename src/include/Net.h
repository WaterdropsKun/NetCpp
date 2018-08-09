#pragma once

#include <opencv2/core.hpp>

#include <iostream>

#include "Function.h"


namespace meng
{
	class Net
	{
	public:
		//Integer vector specifying the number of neurons in each layer including the input and output layers.
		std::vector<int> layer_neuron_num;
		int output_interval = 10;

		std::string activation_function = "sigmoid";
		
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

		//Forward
		void forward();

		//Backward
		void backward();

		//Train, use loss threshold.
		void train(cv::Mat input_, cv::Mat target_, float loss_threshold, bool draw_loss_curve = false);

		//Predict just one sample.
		int predict_one(cv::Mat &input);

		//Test
		void test(cv::Mat &input_, cv::Mat &target_);

		//Save model.
		void save(std::string filename);

		//Load model.
		void load(std::string filename);

	protected:
		//Initialize the weight matrix,if type=0 Gaussian,else uniform.
		void initWeight(cv::Mat &dst, int type, double a, double b);

		//Activation function.
		cv::Mat activationFuntion(cv::Mat &x, std::string func_type);

		//Compute delta error.
		void deltaError();

		//Update weights.
		void updateWeights();

	};

	//Get sample_number samples in XML file, from the start column.
	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start = 0);

	//Draw loss curve
	void draw_curve(cv::Mat& board, std::vector<double> points);

}