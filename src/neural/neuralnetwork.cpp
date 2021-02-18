#include "neuralnetwork.h"
#include "neuron.h"
#include "../misc/functions.h"


NeuralNetwork::NeuralNetwork()
{

}

NeuralNetwork::~NeuralNetwork()
{
    for(Layer* l : _layers)
        delete l;
}


void NeuralNetwork::autogenerate(bool randomize)
{
	connectComplete();
	if(randomize)
		randomizeAllWeights();
	
}

void NeuralNetwork::addLayer(unordered_map<string, double> parameters)
{
	_layers.push_back(new Layer(_layers.size(), this, parameters));
}


void NeuralNetwork::clean()
{
    for(Layer* l : _layers)
        l->clean();
}

void NeuralNetwork::setInput(vector<double> in)
{
	clean();
    for(size_t i=0; i<in.size(); ++i)
        _layers[0]->neurons()[i]->setAccumulated(in[i]);
}

void NeuralNetwork::trigger()
{
	for (Layer* l : _layers)
		l->trigger();
}

vector<double> NeuralNetwork::output()
{
    return (_layers.back())->output();
}

string NeuralNetwork::outputString()
{
	string s;
	for (size_t i = 0; i < (_layers.back())->output().size(); i++)
	{
		double out = (_layers.back())->output()[i];
		s += to_string(out) + " ";
	}
	return s;
}

void NeuralNetwork::connectComplete()
{
    for(size_t i_layer = 0; i_layer < _layers.size()-1; ++i_layer)
		_layers[i_layer]->connectComplete(_layers[i_layer+1]);
}

void NeuralNetwork::alterWeights(const vector<vector<vector<double> > >& weights)
{
	for (size_t i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
		_layers[i_layer]->alterWeights(weights[i_layer]);
}

void NeuralNetwork::shiftBackWeights(const vector<vector<vector<double> > >& weights)
{
	for (int i_layer = _layers.size() - 1; i_layer >= 0; --i_layer)
		if(weights[i_layer].size() != 0)
			_layers[i_layer]->shiftBackWeights(weights[i_layer]);
}

vector<vector<vector<double*>>> NeuralNetwork::getWeights()
{
	vector<vector<vector<double*>>> w;
	w.reserve(_layers.size() - 1);
	for (size_t i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
		w.push_back(std::move(_layers[i_layer]->getWeights()));
	return std::move(w);
}

vector<vector<vector<Edge*>>> NeuralNetwork::getEdges()
{
	vector<vector<vector<Edge*>>> w;
	w.reserve(_layers.size() - 1);
	for (size_t i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
		w.push_back(std::move(_layers[i_layer]->getEdges()));
	return std::move(w);
}

void NeuralNetwork::randomizeAllWeights()
{
	for(size_t i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
		_layers[i_layer]->randomizeAllWeights(RAND_MAX_WEIGHT); //random weights from -RAND_MAX_WEIGHT to RAND_MAX_WEIGHT
}

double NeuralNetwork::loss(const vector<double>& in, const vector<double>& out)
{
	double sum = 0;
	auto out_exp = predict(in);
	if (_layers.back()->getParameters().at("activation") == ActivationFunction::SIGMOID)
		for (size_t i = 0; i < out.size(); ++i)
			sum += 0.5 * (out[i] - out_exp[i]) * (out[i] - out_exp[i]);
	if (_layers.back()->getParameters().at("activation") == ActivationFunction::LINEAR)
		for (size_t i = 0; i < out.size(); ++i)
			sum += (out[i] - out_exp[i]) * (out[i] - out_exp[i]);
	return sum;
}

double NeuralNetwork::loss(const vector<vector<double>*>& ins, const vector<vector<double>*>& outs)
{
	double sum = 0;
	for (size_t i = 0; i < ins.size(); i++)
	{
		sum += loss(*ins[i], *outs[i]);
	}
	return sum / ins.size() ;
}

string NeuralNetwork::toString()
{
    string s = "NeuralNetwork";
    s.push_back('\n');
    for(Layer* l : _layers)
        s += l->toString();
    return s;
}


void NeuralNetwork::shiftWeights(float percentage_of_range)
{
	float range = percentage_of_range * (RAND_MAX_WEIGHT + RAND_MAX_WEIGHT); //distance entre min et max des poids
	for(Layer* l : _layers)
		l->shiftWeights(range);
}


vector<double> NeuralNetwork::predict(const vector<double>& in)
{
	setInput(in);
	trigger();
	return output();
}

double NeuralNetwork::predictAllForScore(const Dataset& dataset, Datatype d,  int limit)
{
	if (limit == 0)
		return 1;
	double s = 0;

	//Sans limite explicite, on score toutes les données
	if (limit == -1)
		for (size_t i = 0; i < dataset.getIns(d).size(); i++)
			s += distanceVector(predict(*dataset.getIns(d)[i]), *dataset.getOuts(d)[i]);
	//Sinon on prend "limit" données
	else
		for (int i = 0; i < limit; i++)
		{
			int r = rand() % dataset.getIns(d).size();
			s += distanceVector(predict(*dataset.getIns(d)[r]), *dataset.getOuts(d)[r]);
		}

	//On moyenne le score
	if (limit == -1)
		s /= dataset.getIns(d).size();
	else
		s /= limit;
	return s;
}

vector<Layer*> NeuralNetwork::getLayers()
{
	return _layers;
}
