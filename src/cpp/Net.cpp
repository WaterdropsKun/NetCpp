#include "Net.h"

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

namespace meng
{

	void Net::initNet(std::vector<int> layer_neuron_num_)
	{
		layer_neuron_num = layer_neuron_num_;

		//Generate every layer.
		layer.resize(layer_neuron_num.size());
		for (int i = 0; i < layer.size(); i++)
		{
			layer[i].create(layer_neuron_num[i], 1, CV_32FC1);
		}
		std::cout << "Generate layers successfully!" << std::endl;

		//Generate every weights matrix and bias.
		weights.resize(layer.size() - 1);
		bias.resize(layer.size() - 1);
		for (int i = 0; i < (layer.size() - 1); i++)   //Debug
		{
			weights[i].create(layer[i + 1].rows, layer[i].rows, CV_32FC1);
			bias[i] = cv::Mat::zeros(layer[i + 1].rows, 1, CV_32FC1);
		}
		std::cout << "Generate weights matrices and bias successfully!" << std::endl;

		std::cout << "Initialize Net done!" << std::endl;
	}

	void Net::initWeights(int type /*= 0*/, double a /*= 0.*/, double b /*= 0.1*/)
	{
		//Initialize the weights matrix.
		for (int i = 0; i < weights.size(); i++)
		{
			initWeight(weights[i], 0, 0., 0.1);
		}
	}

	void Net::initBias(cv::Scalar& bias_)
	{
		for (int i = 0; i < bias.size(); i++)
		{
			//Scalar类型变量值可以直接赋值给Mat类型变量
			bias[i] = bias_;
		}
	}

	void Net::forward()
	{
		for (int i = 0; i < layer_neuron_num.size() - 1; i++)
		{
			cv::Mat product = weights[i] * layer[i] + bias[i];
			layer[i + 1] = activationFuntion(product, activation_function);   //Debug
		}
		calcLoss(layer[layer.size() - 1], target, output_error, loss);
	}

	void Net::backward()
	{
		deltaError();
		updateWeights();
	}

	void Net::train(cv::Mat input_, cv::Mat target_, float loss_threshold, bool draw_loss_curve /*= false*/)
	{
		if (input_.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return;
		}

		std::cout << "Train begin!" << std::endl;

		cv::Mat sample;
		if (input_.rows == (layer[0].rows) && input_.cols == 1)
		{
			
		}
		else if (input_.rows == (layer[0].rows) && input_.cols > 1)
		{
			double batch_loss = loss_threshold + 0.01;
			int epoch = 0;

			while (batch_loss > loss_threshold)
			{
				batch_loss = 0;
				for (int i = 0; i < input_.cols; i++)
				{
					target = target_.col(i);
					sample = input_.col(i);
					layer[0] = sample;

					forward();
					batch_loss += loss;
					backward();					
				}

				loss_vec.push_back(batch_loss);
				if (loss_vec.size() >= 2 && draw_loss_curve)
				{
					draw_curve(board, loss_vec);
				}

				epoch++;
				if (epoch % output_interval == 0)
				{
					std::cout << "Number of epoch: " << epoch << std::endl;
					std::cout << "Loss sum: " << batch_loss << std::endl;
				}
				if (epoch % 100 == 0)
				{
					learning_rate *= fine_tune_factor;

				}
			}

			std::cout << std::endl << "Number of epoch: " << epoch << std::endl;
			std::cout << "Loss sum: " << batch_loss << std::endl;
			std::cout << "Train successfully!" << std::endl;
		}
		else
		{
			std::cout << "Rows of input don't match the number of input!" << std::endl;
		}
	}

	int Net::predict_one(cv::Mat &input_)
	{
		if (input_.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return -1;
		}

		if (input_.rows == (layer[0].rows) && input_.cols == 1)
		{
			layer[0] = input_;

			forward();

			cv::Point predict_maxLoc;
			minMaxLoc(layer[layer.size() - 1], NULL, NULL, NULL, &predict_maxLoc, cv::noArray());
			return predict_maxLoc.y;
		}
		else
		{
			std::cout << "Please give one sample alone and ensure input.rows == layer[0].rows" << std::endl;
			return -1;
		}
	}

	void Net::test(cv::Mat &input_, cv::Mat &target_)
	{
		if (input_.empty())
		{
			std::cout << "Input is empty!" << std::endl;
			return;
		}
		std::cout << std::endl << "Predict begin!" << std::endl;

		if (input_.rows == layer[0].rows && input_.cols == 1)
		{

		}
		else if (input_.rows == (layer[0].rows) && input_.cols > 1)
		{
			double loss_sum = 0;
			int right_num = 0;

			cv::Mat sample;
			for (int i = 0; i < input_.cols; i++)
			{
				sample = input_.col(i);
				int predict_number = predict_one(sample);
				loss_sum += loss;

				target = target_.col(i);
				cv::Point target_maxLoc;
				minMaxLoc(target, NULL, NULL, NULL, &target_maxLoc, cv::noArray());
				int target_number = target_maxLoc.y;

				std::cout << "Test sample: " << i << "   " << "Predict: " << predict_number << std::endl;
				std::cout << "Test sample: " << i << "   " << "Target: " << target_number << std::endl << std::endl;

				if (predict_number == target_number)
				{
					right_num++;
				}
			}

			accuracy = (double)right_num / input_.cols;
			std::cout << "Loss sum: " << loss_sum << std::endl;
			std::cout << "accuracy_mk: " << accuracy << std::endl;
		}
		else
		{
			std::cout << "Rows of input don't match the number of input!" << std::endl;
			return;
		}
	}

	void Net::save(std::string filename)
	{
		cv::FileStorage model(filename, cv::FileStorage::WRITE);
		model << "layer_neuron_num" << layer_neuron_num;
		model << "learning_rate" << learning_rate;
		model << "activation_function" << activation_function;

		for (int i = 0; i < weights.size(); i++)
		{
			std::string weight_name = "weight_" + std::to_string(i);
			model << weight_name << weights[i];

			std::string bias_name = "bias_" + std::to_string(i);
			model << bias_name << bias[i];
		}

		model.release();
	}

	void Net::load(std::string filename)
	{
		cv::FileStorage fs;
		fs.open(filename, cv::FileStorage::READ);

		fs["layer_neuron_num"] >> layer_neuron_num;
		fs["learning_rate"] >> learning_rate;
		fs["activation_function"] >> activation_function;

		initNet(layer_neuron_num);

		for (int i = 0; i < weights.size(); i++)
		{
			std::string weight_name = "weight_" + std::to_string(i);
			fs[weight_name] >> weights[i];

			std::string bias_name = "bias_" + std::to_string(i);
			fs[bias_name] >> bias[i];
		}

		fs.release();
	}

	void Net::initWeight(cv::Mat &dst, int type, double a, double b)
	{
		if (type == 0)
		{
			randn(dst, a, b);
		}
		else
		{
			randu(dst, a, b);
		}
	}

	cv::Mat Net::activationFuntion(cv::Mat &x, std::string func_type)
	{
		activation_function = func_type;

		cv::Mat fx;
		if (func_type == "sigmoid")
		{
			fx = sigmoid(x);
		}

		return fx;
	}

	void Net::deltaError()
	{
		delta_err.resize(layer.size() - 1);
		for (int i = delta_err.size() - 1; i >= 0; i--)
		{
			delta_err[i].create(layer[i + 1].size(), layer[i + 1].type());

			cv::Mat dx = derivativeFuntion(layer[i + 1], activation_function);

			//Output layer delta error
			if (i == delta_err.size() - 1)
			{
				delta_err[i] = dx.mul(output_error);
			}
			//Hidden layer delta error
			else
			{
				delta_err[i] = dx.mul((weights[i + 1]).t() * delta_err[i + 1]);   //Debug				
			}
		}		
	}

	void Net::updateWeights()
	{
		for (int i = 0; i < weights.size(); i++)
		{
			cv::Mat delta_weight = learning_rate * (delta_err[i] * layer[i].t());
			cv::Mat delta_bias = learning_rate * delta_err[i];

			weights[i] = weights[i] + delta_weight;
			bias[i] = bias[i] + delta_bias;
		}
	}

	void get_input_label(std::string filename, cv::Mat& input, cv::Mat& label, int sample_num, int start /*= 0*/)
	{
		cv::FileStorage fs;
		fs.open(filename, cv::FileStorage::READ);

		cv::Mat input_, target_;
		fs["input"] >> input_;
		fs["target"] >> target_;

		fs.release();

		input = input_(cv::Rect(start, 0, sample_num, input_.rows));
		label = target_(cv::Rect(start, 0, sample_num, target_.rows));
	}

	void draw_curve(cv::Mat& board, std::vector<double> points)
	{
		cv::Mat board_(620, 1000, CV_8UC3, cv::Scalar::all(200));
		board = board_;

		cv::line(board, cv::Point(0, 550), cv::Point(1000, 550), cv::Scalar(0, 0, 0), 2);
		cv::line(board, cv::Point(50, 0), cv::Point(50, 1000), cv::Scalar(0, 0, 0), 2);

		for (size_t i = 0; i < points.size() - 1; i++)
		{
			cv::Point pt1(50 + i * 2, (int)(548 - points[i]));
			cv::Point pt2(50 + i * 2 + 1, (int)(548 - points[i + 1]));
			cv::line(board, pt1, pt2, cv::Scalar(0, 0, 255), 2);

			if (i > 1000)
			{
				return;
			}
		}

		cv::imshow("Loss", board);
		cv::waitKey(1);
	}
}


