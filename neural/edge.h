#ifndef EDGE_H
#define EDGE_H

#include "mainheader.h"

extern double LEARNING_RATE;
extern double SHIFT_LIMIT;

//Edge between two neurons
class Edge
{
public:
    Edge(Neuron* n, Neuron* start, double w );

	Neuron* neuron() const;

	Neuron* neuronb() const;

    double weight() const;

    void propagate(double neuron_output);

    void alterWeight(double w);

	void shiftWeight(double dw);

	double getLastShift() const;

	double backprogationMemory() const;

	void setBackpropagationMemory(double v);

private:
	Neuron* _n = nullptr;
	Neuron* _nb = nullptr;
    double _w = 0.0;
	double _last_shift = 0;

	double _backpropagation_memory;

};

#endif // EDGE_H
