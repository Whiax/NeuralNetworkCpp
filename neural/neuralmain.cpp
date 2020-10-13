#include "mainheader.h"
#include "neuralnetwork.h"
#include "functions.h"
#include<time.h>
#include  <numeric>

#include <fstream>




double LEARNING_RATE = 0.500;
double SHIFT_LIMIT = 1;




//Main function
int main(int argc, char *argv[])
{
	












	srand(20);

	NeuralNetwork n;
	n.addLayer({ { "type", LayerType::INPUT },{ "size",2 }});
	for (size_t i = 0; i < 1; i++)
		n.addLayer({ { "type", LayerType::STANDARD },{ "size",2} ,{ "activation",ActivationFunction::SIGMOID } });
	n.addLayer({ { "type", LayerType::OUTPUT},{ "size",2} ,{ "activation",ActivationFunction::SIGMOID } });
	n.autogenerate();

	n.alterWeights({ {{0.15,0.25},{0.20,0.30},{0.35,0.35}},{ { 0.40,0.50 } ,{ 0.45,0.55 } ,{ 0.60, 0.60 } } });
	vector<vector<double>> data = { { 0.05,0.10 },{ 0.01,0.99 } };
	for (size_t i = 0; i < 100000; i++)
	{
		//cout << "loss:" << n.loss({ 0.05,0.10 }, { 0.01,0.99 }) << endl;
		cout << n.toString() << endl;
		n.backpropagate({ &data[0] }, { &data[1] });
		getchar();
	}
	return 0;
}
