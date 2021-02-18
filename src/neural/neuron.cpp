#include "neuron.h"
#include <algorithm>
#include "neuralnetwork.h"

Neuron::Neuron(int id_neuron, Layer* layer, ActivationFunction function, bool is_bias):
    _id_neuron(id_neuron),
    _layer(layer),
	_activation_function(function),
	_is_bias(is_bias)
{	
}

Neuron::~Neuron()
{
    for(Edge* e : _next)
        delete e;
}

void Neuron::trigger()
{
	for (Edge* e : _next)
	{
		//cout << this->getNeuronId() << "->" << e->neuron()->getNeuronId() << " " << output() << "*" << e->weight() << "=" << output()*e->weight() << "("<< outputRaw() << ")" << endl;
		e->propagate(output());
	}

}

double Neuron::in()
{
    return _accumulated;
}

double Neuron::output()
{
	if (_is_bias)
		return 1;
    if(_layer->getType() == LayerType::INPUT)
        return outputRaw();

	//return random(-10, 10);
	if (_activation_function == ActivationFunction::LINEAR)
		return _accumulated;
	if(_activation_function == ActivationFunction::RELU)
		return relu(_accumulated);
	if (_activation_function == ActivationFunction::SIGMOID)
		return sigmoid(_accumulated);
	return outputRaw();
}

double Neuron::outputDerivative()
{
	if (_activation_function == ActivationFunction::LINEAR)
		return 1;
	if (_activation_function == ActivationFunction::RELU)
		return relu_derivative(output());
	if (_activation_function == ActivationFunction::SIGMOID)
		return sigmoid_derivative(outputRaw());
	return _accumulated;
}

double Neuron::outputRaw()
{
    return _accumulated;
}

void Neuron::clean()
{
    setAccumulated(0);
}

void Neuron::addAccumulated(double v)
{
	//cout << this->_layer->getId() << ":" << this->getNeuronId() << " added " << v << " on " << _accumulated << endl;

    setAccumulated(_accumulated + v);
}

void Neuron::addNext(Neuron *n)
{
    _next.push_back(new Edge(n, this, random(-5, 5)));
	n->addPrevious(_next.back());
}

void Neuron::addPrevious(Edge* e)
{
	_previous.push_back(e);
}

int Neuron::getNeuronId() const
{
    return _id_neuron;
}

void Neuron::setAccumulated(double v)
{

    _accumulated = v;
}

void Neuron::alterWeights(const vector<double>& weights)
{
    for(size_t i_edge=0; i_edge < weights.size(); ++i_edge)
        _next[i_edge]->alterWeight(weights[i_edge]);
}

vector<double*> Neuron::getWeights()
{
	vector<double*> w;
	w.reserve(_next.size());
	for (size_t i_edge = 0; i_edge < _next.size(); ++i_edge)
		w.push_back(_next[i_edge]->weightP());
	return std::move(w);
}

vector<Edge*> Neuron::getEdges()
{
	vector<Edge*> w;
	w.reserve(_next.size());
	for (size_t i_edge = 0; i_edge < _next.size(); ++i_edge)
		w.push_back(_next[i_edge]);
	return std::move(w);
}

void Neuron::randomizeAllWeights(double abs_value)
{
    for(Edge* e : _next)
		e->alterWeight(random(-abs_value, abs_value));
}


string Neuron::toString()
{
    string weights;
    for(Edge* e : _next)
        weights.append( to_string(e->weight()) + ",");
    string str =  "[" +  to_string(_layer->getId()) + "," + to_string(_id_neuron) + "]" +"("+ weights +")";
    return str;
}

void Neuron::shiftWeights(float range)
{
	for (Edge* e : _next)
		e->alterWeight(e->weight() + random(-range, range));
}

void Neuron::shiftBackWeights(const vector<double>& w)
{
	for (size_t i = 0; i < _previous.size(); i++)
		_previous[i]->shiftWeight(w[i]);
}

//gradient descent
vector<double> Neuron::getBackpropagationShifts(const vector<double>& target)
{
	vector<double> dw(_previous.size(),0);
	if (_layer->getType() == LayerType::OUTPUT)
	{
		double d0 = output();
		double d1 = output() - target[this->getNeuronId()];
		double d2 = outputDerivative();
		for (size_t i = 0; i < _previous.size(); ++i)
		{
			dw[i] = (-d1*d2*_previous[i]->neuronb()->output());
			_previous[i]->setBackpropagationMemory(d1*d2);
		}
		//cout << _layer->getId() << " " << d1 << " " << d2 << " " << d3 << " " << d1*d2*d3 << endl;

	}
	else 
	{
		double d = 0;
		for (size_t i = 0; i < _next.size(); i++)
			d += _next[i]->backpropagationMemory() * _next[i]->weight();
		d *= outputDerivative();
		for (size_t i = 0; i < _previous.size(); i++)
		{
			_previous[i]->setBackpropagationMemory(d);
			dw[i] = -d * _previous[i]->neuronb()->output();
		}
	}
	return dw;
}

bool Neuron::isBias() const
{
	return _is_bias;
}
