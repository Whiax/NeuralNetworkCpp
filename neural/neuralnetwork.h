#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "mainheader.h"
#include "functions.h"
#include "layer.h"
#include <unordered_map>

#define RAND_MAX_WEIGHT 1

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

	vector<vector<vector<double>>> getWeights();

    void randomizeAllWeights();

    double loss(const vector<double>& in, const vector<double>& out);

	double loss(const vector<vector<double>*>& ins, const vector<vector<double>*>& outs);

    string toString();

	void shiftWeights(float percentage_of_range);

	vector<double> predict(const vector<double>& in);

	vector<vector<vector<double>>> getBackpropagationShifts(const vector<double>& in, const vector<double>& out);

	void backpropagate(const vector<vector<double>*>& ins, const vector<vector<double>*>& outs);

	vector<Layer*> getLayers();

private:
    vector<Layer*> _layers;
	double _fitness;

	vector<unordered_map<string,double>> _configuration;
};

#endif // NEURALNETWORK_H
