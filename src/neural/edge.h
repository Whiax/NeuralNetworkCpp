#ifndef EDGE_H
#define EDGE_H


#include <iostream>

#include <vector>
#include <limits>

class NeuralNetwork;
class Neuron;
class Layer;
using namespace std;

typedef unsigned int uint;


extern double LEARNING_RATE;

//Edge between two neurons
class Edge
{
public:
    Edge(Neuron* n, Neuron* start, double w );

	Neuron* neuron() const;

	Neuron* neuronb() const;

	double weight();

	double* weightP();

    void propagate(double neuron_output);

    void alterWeight(double w);

	void shiftWeight(double dw);

	double getLastShift() const;

	void resetLastShift();

	double backpropagationMemory() const;

	void setBackpropagationMemory(double v);

public:
	Neuron* _n = nullptr;
	Neuron* _nb = nullptr;
    double _w = 0.0;
	double _last_shift = 0;

	double _backpropagation_memory;

};

#endif // EDGE_H
