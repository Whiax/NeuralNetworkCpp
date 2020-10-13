#include "optimizer.h"
#include <thread>


Optimizer::Optimizer()
{
	LEARNING_RATE = 1;
}


Optimizer::~Optimizer()
{
}

void Optimizer::minimize()
{
	cout << "not implemented" << endl;
}

void Optimizer::minimizeThread()
{
	/*vector<thread> t;
	for (size_t i = 0; i < 10; i++)
		t.push_back(move(std::thread(&Optimizer::minimize, this)));
	for (size_t i = 0; i < t.size(); i++)
		t[i].join();*/

	std::thread t(&Optimizer::minimize, this);
	t.join();
}

void Optimizer::setDataset(Dataset* dataset)
{
	_d = dataset;
}

void Optimizer::setNeuralNetwork(NeuralNetwork* net)
{
	_n = net;
}


double Optimizer::getScore(Datatype d, int limit)
{
	return _n->predictAllForScore(*_d,d, limit);
}

