#include "neuralnetwork.h"
#include "neuron.h"

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
    for(int i_layer = 0; i_layer < _layers.size()-1; ++i_layer)
		_layers[i_layer]->connectComplete(_layers[i_layer+1]);
}

void NeuralNetwork::alterWeights(const vector<vector<vector<double> > >& weights)
{
	for (int i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
		_layers[i_layer]->alterWeights(weights[i_layer]);
}

void NeuralNetwork::shiftBackWeights(const vector<vector<vector<double> > >& weights)
{
	for (int i_layer = _layers.size() - 1; i_layer >= 0; --i_layer)
		if(weights[i_layer].size() != 0)
			_layers[i_layer]->shiftBackWeights(weights[i_layer]);
}

vector<vector<vector<double>>> NeuralNetwork::getWeights()
{
	vector<vector<vector<double>>> w;
	for(int i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
		w.push_back(std::move(_layers[i_layer]->getWeights()));
	return std::move(w);
}

void NeuralNetwork::randomizeAllWeights()
{
	for(int i_layer = 0; i_layer < _layers.size() - 1; ++i_layer)
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

vector<vector<vector<double>>> NeuralNetwork::getBackpropagationShifts(const vector<double>& in, const vector<double>& out)
{
	vector<vector<vector<double>>> dw(_layers.size());
	auto out_exp = predict(in);
	for (int i = _layers.size() - 1; i >= 1; --i)
	{
		auto _dw = move(_layers[i]->getBackpropagationShifts(out));
		dw[_layers[i]->getId()] = _dw;
	}
	return move(dw);
	
}

void  NeuralNetwork::backpropagate(const vector<vector<double>*>& ins, const vector<vector<double>*>& outs)
{
	vector<vector<vector<double>>> dw(_layers.size());
	bool is_init = false;
	for (size_t i = 0; i < ins.size(); i++)
	{
		auto in = ins[i];
		auto out = outs[i];
		auto _dw = getBackpropagationShifts(*in, *out);
		if (!is_init)
		{
			for (size_t j = 0; j < _dw.size(); j++)
			{
				dw[j].resize(_dw[j].size());
				for (size_t k = 0; k < _dw[j].size(); k++)
					dw[j][k].resize(_dw[j][k].size(),0);
			}
		}
		for (size_t j = 0; j < _dw.size(); j++)
			for (size_t k = 0; k < _dw[j].size(); k++)
				for (size_t l = 0; l < _dw[j][k].size(); l++)
					dw[j][k][l] += _dw[j][k][l]; //edge l, neuron k, layer j
	}
	for (size_t j = 0; j < dw.size(); j++)
		for (size_t k = 0; k < dw[j].size(); k++)
			for (size_t l = 0; l < dw[j][k].size(); l++)
				dw[j][k][l] /= ins.size();

	shiftBackWeights(dw);
}


vector<Layer*> NeuralNetwork::getLayers()
{
	return _layers;
}
