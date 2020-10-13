#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "../misc/functions.h"
#include "layer.h"
#include "../dataset/dataset.h"
#include <unordered_map>

#define RAND_MAX_WEIGHT 1

#include <iostream>

#include <vector>
#include <limits>

class NeuralNetwork;
class Neuron;
class Layer;
using namespace std;

typedef unsigned int uint;


class NeuralNetwork
{
public:
	NeuralNetwork();

    ~NeuralNetwork();

	void autogenerate(bool randomize = true);

	void addLayer(unordered_map<string, double> parameters);

    void clean();

	void setInput(vector<double> in);

    void trigger();

    vector<double> output();

	string outputString();

    void connectComplete();

	void alterWeights(const vector<vector<vector<double>>>& weights);

	void shiftBackWeights(const vector<vector<vector<double>>>& weights);

	vector<vector<vector<double*>>> getWeights();

	vector<vector<vector<Edge*>>> getEdges();

    void randomizeAllWeights();

    double loss(const vector<double>& in, const vector<double>& out);

	double loss(const vector<vector<double>*>& ins, const vector<vector<double>*>& outs);

    string toString();

	void shiftWeights(float percentage_of_range);

	vector<double> predict(const vector<double>& in);

	double predictAllForScore(const Dataset& dataset, Datatype d = TEST, int limit=-1);

	double predictPartialForScore(const Dataset& dataset);

	vector<Layer*> getLayers();





public:
    vector<Layer*> _layers;
	double _fitness;

	vector<unordered_map<string,double>> _configuration;
};

#endif // NEURALNETWORK_H
