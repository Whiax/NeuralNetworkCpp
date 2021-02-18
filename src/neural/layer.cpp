#include "layer.h"

Layer::Layer(int id_layer, NeuralNetwork* net, unordered_map<string, double> parameters)
{
    _id_layer = id_layer;
	_net = net;
	_parameters = parameters;
	_type = static_cast<LayerType>(static_cast<int>(parameters["type"])); //erk
	_activation = static_cast<ActivationFunction>(static_cast<int>(_parameters["activation"]));
	initLayer();
}

Layer::~Layer()
{
    for(Neuron* n : _neurons)
        delete n;
    _neurons.clear();
}

int Layer::getId() const
{
	return _id_layer;
}

void Layer::initLayer()
{
	_neurons.clear();

	if (_type == LayerType::STANDARD || _type == LayerType::INPUT)
	{
		_parameters["size"] += 1; //bias for the next layer
		_neurons.reserve(static_cast<int>(_parameters["size"]));
		for (int i_neuron = 0; i_neuron < _parameters["size"]; ++i_neuron)
			_neurons.push_back(new Neuron(i_neuron, this, _activation, i_neuron == _neurons.capacity()-1));
	}
	else if (_type == LayerType::OUTPUT)
	{
		_neurons.reserve(static_cast<int>(_parameters["size"]));
		for (int i_neuron = 0; i_neuron < _parameters["size"]; ++i_neuron)
			_neurons.push_back(new Neuron(i_neuron, this, _activation));
	}
}

void Layer::clean()
{
    for(Neuron* n : _neurons)
        n->clean();
}

void Layer::trigger()
{
    for(Neuron* n : _neurons)
        n->trigger();
}


void Layer::connectComplete(Layer *next)
{
    for(Neuron* n1 : _neurons)
        for(Neuron* n2 : next->_neurons)
			if(!n2->isBias())
				n1->addNext(n2);
}

vector<double> Layer::output()
{
    vector<double> outputs;
    outputs.reserve(_neurons.size());
    for(Neuron* n : _neurons)
        outputs.push_back(n->output());
    return std::move(outputs);

}

const vector<Neuron *>& Layer::neurons() const
{
    return _neurons;
}

void Layer::alterWeights(const vector<vector<double> >& weights)
{
    for(size_t i_neuron=0;i_neuron < weights.size(); ++i_neuron)
        _neurons[i_neuron]->alterWeights(weights[i_neuron]);
}

void Layer::shiftBackWeights(const vector<vector<double> >& weights)
{
	for (size_t i_neuron = 0; i_neuron < _neurons.size(); ++i_neuron)
		_neurons[i_neuron]->shiftBackWeights(weights[i_neuron]);
}

vector<vector<double*> > Layer::getWeights()
{
	vector<vector<double*>> w;
	w.reserve(_neurons.size());
	for (size_t i_neuron = 0; i_neuron < _neurons.size(); ++i_neuron)
		w.push_back(std::move(_neurons[i_neuron]->getWeights()));
	return std::move(w);
}

vector<vector<Edge*> > Layer::getEdges()
{
	vector<vector<Edge*>> w;
	w.reserve(_neurons.size());
	for (size_t i_neuron = 0; i_neuron < _neurons.size(); ++i_neuron)
		w.push_back(std::move(_neurons[i_neuron]->getEdges()));
	return std::move(w);
}

void Layer::randomizeAllWeights(double abs_value)
{
    for(Neuron* neuron : _neurons)
        neuron->randomizeAllWeights(abs_value);
}

string Layer::toString()
{
    string str = "layer:" + to_string(_id_layer) + "\n";
    for(Neuron* neuron : _neurons)
        str += neuron->toString() + "\n";
    return str;
}

void Layer::shiftWeights(float range)
{
	for(Neuron* neuron : _neurons)
		neuron->shiftWeights(range);
}

const unordered_map<string, double>& Layer::getParameters() const
{
	return _parameters;
}

vector<vector<double>> Layer::getBackpropagationShifts(const vector<double>& target)
{
	vector<vector<double>> dw(_neurons.size());
	for (size_t i = 0; i < _neurons.size(); i++)
	{
		Neuron* n = _neurons[i];
		dw[i] = n->getBackpropagationShifts(target);
	}
	return dw;
}

LayerType Layer::getType() const
{
	return _type;
}

ActivationFunction Layer::getActivation() const
{
	return _activation;
}
