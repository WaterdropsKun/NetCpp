#include "Net.h"

using namespace std;
using namespace cv;
using namespace meng;


int main1(int argc, char *argv[])
{
	//Set neuron number of every layer.
	vector<int> layer_neuron_num = { 784, 100, 10 };

	//Initialize Net.
	Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0.0, 0.01);
	net.initBias(Scalar(0.05));

	//Get test samples and test samples.
	Mat input, label, test_input, test_label;
	int sample_number = 200;
	get_input_label("./data/input_label_1000.xml", input, label, sample_number);
	get_input_label("./data/input_label_1000.xml", test_input, test_label, 200, 800);

	//0£∫—µ¡∑,1£∫≤‚ ‘
	int flag = 1;
	if (flag == 0)
	{
		//Set loss threshold, learning rate and activation function.
		float loss_threshold = 0.5;
		net.learning_rate = 0.3;
		net.output_interval = 2;   //Debug
		net.activation_function = "sigmoid";

		//Train and draw the loss curve and test the trained net
		net.train(input, label, loss_threshold, true);
		net.test(test_input, test_label);

		//Save model.
		net.save("./model/model_sigmoid_800_200.xml");
	}
	if (flag == 1)
	{
		net.load("./model/model_sigmoid_800_200.xml");

		net.test(test_input, test_label);
	}	

	getchar();
	return 0;

}