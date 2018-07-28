#include "Net.h"

using namespace std;
using namespace cv;
using namespace meng;




int main(int argc, char *argv[])
{
	//Set neuron number of every layer.
	vector<int> layer_neuron_num = { 784, 100, 10 };

	//Initialize Net.
	Net net;
	net.initNet(layer_neuron_num);
	net.initWeights(0, 0.0, 0.01);
	net.initBias(Scalar(0.05));

	getchar();
	return 0;

}