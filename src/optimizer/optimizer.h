#pragma once

#include "../dataset/dataset.h"
#include "../neural/neuralnetwork.h"


class Optimizer
{
public:
	Optimizer();
	~Optimizer();

	virtual void minimize();

	void setNeuralNetwork(NeuralNetwork* net);

	void setDataset(Dataset* dataset);

	double getScore(Datatype d, int limit = -1);

	void minimizeThread();

protected:
	NeuralNetwork* _n;

	Dataset* _d;

};

