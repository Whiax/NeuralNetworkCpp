
#include "neural/neuralnetwork.h"
#include "misc/functions.h"
#include "optimizer/backpropagation.h"
#include "optimizer/shakingtree.h"
#include "dataset/dataset.h"

#include <ctime>
#include <numeric>
#include <fstream>


#include <iostream>

#include <vector>
#include <limits>
#include <thread>

class NeuralNetwork;
class Neuron;
class Layer;
using namespace std;

typedef unsigned int uint;



double LEARNING_RATE = 0.5000;




//Main function
int main(int argc, char *argv[])
{
	clock_t t;
	srand(time(0));
	//ofstream logs("logs_minimizeBasicLarger2.txt");

	NeuralNetwork n;
	n.addLayer({ { "type", LayerType::INPUT },{ "size",2 }});
	for (size_t i = 0; i < 5; i++)
		n.addLayer({ { "type", LayerType::STANDARD },{ "size",10} ,{ "activation",ActivationFunction::SIGMOID } });
	n.addLayer({ { "type", LayerType::OUTPUT},{ "size",1} ,{ "activation",ActivationFunction::SIGMOID } });
	n.autogenerate();

	Dataset data("data1000.txt");
	data.split(0.8);

	
	/*Shakingtree opt;
	opt.setNeuralNetwork(&n);
	opt.setDataset(&data);
	opt.mapParameters();*/

	Backpropagation opt;
	opt.setBatchSize(60);
    LEARNING_RATE = 0.5;
	opt.setNeuralNetwork(&n);
	opt.setDataset(&data);

	double mintest = 1;
	int i = 0;
	int validate_every = 10;
	int n_iteration = 50000;
	t = clock();
	while (i < n_iteration)
	{
		opt.minimize();

		if (i % validate_every == 0)
		{
			double strain = n.predictAllForScore(data, TRAIN);
			double stest = n.predictAllForScore(data);
			mintest = stest < mintest ? stest : mintest;
			cout << "it:" << i << '\t' << "    test_score:" << stest << "    train_score:" << strain << "   (best_test_score : " << mintest << ")" << endl;

			float delta_t = (clock() - t) / 1000.0;
			//logs << i << ";" << delta_t << ";" << stest << ";" << strain << ";" << mintest << endl;
		}
		if (i % 500 == 0)
		{
			LEARNING_RATE = LEARNING_RATE * 0.9;
			cout << LEARNING_RATE << endl;
		}
		i++;
	}

	getchar();
    return 0;
}
