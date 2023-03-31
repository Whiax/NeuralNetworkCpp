
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
	//Init random and logs if required
	srand(uint(time(0)));
	//ofstream logs("logs.txt");



	//Neural network definition
	NeuralNetwork n;
	size_t n_hidden_layer = 5;
	n.addLayer({ { "type", LayerType::INPUT },{ "size",2 }});
	for (size_t i = 0; i < n_hidden_layer; i++)
		n.addLayer({ { "type", LayerType::STANDARD },{ "size",10} ,{ "activation",ActivationFunction::SIGMOID } });
	n.addLayer({ { "type", LayerType::OUTPUT},{ "size",1} ,{ "activation",ActivationFunction::SIGMOID } });
	n.autogenerate();


	//Create the dataset
	Dataset data("data1000.txt");
	data.split(0.8);


	//Create the optimizer
	/*Shakingtree opt;
	opt.setNeuralNetwork(&n);
	opt.setDataset(&data);
	opt.mapParameters();*/

	Backpropagation opt;
	opt.setBatchSize(60);
        LEARNING_RATE = 0.5;
	opt.setNeuralNetwork(&n);
	opt.setDataset(&data);



	//Init the main training loop (nb: the goal is to lower the score, score = loss)
	double lr_reduce_amplitude = 0.9;
	int lr_reduce_schedule = 500;
	int n_iteration = 50000;
	int validate_every = 10;
	double mintest = 1;
	int i = 0;
	clock_t t = clock();
	while (i < n_iteration)
	{
		//For Backpropagation: The optimizer reads a batch, pass it in the neural network, computes and apply gradients
		//For ShakingTree: The behaviour depends but the overall idea is to try random parameters and keep the good ones
		opt.minimize();

		//Evaluate the score on test data
		if (i % validate_every == 0)
		{
			double strain = n.predictAllForScore(data, TRAIN);
			double stest = n.predictAllForScore(data);
			mintest = stest < mintest ? stest : mintest;
			cout << "it:" << i << '\t' << "    test_score:" << stest << "    train_score:" << strain << "   (best_test_score : " << mintest << ")" << endl;

			double delta_t = (clock() - t) / 1000.0;
			//logs << i << ";" << delta_t << ";" << stest << ";" << strain << ";" << mintest << endl;
		}

		//Reduce learning rate
		if (i % lr_reduce_schedule == 0)
		{
			LEARNING_RATE = LEARNING_RATE * lr_reduce_amplitude;
			cout << LEARNING_RATE << endl;
		}
		i++;
	}

	getchar();
    return 0;
}
