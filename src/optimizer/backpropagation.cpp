#include "Backpropagation.h"


void Backpropagation::setLearningRate(double lr)
{
	LEARNING_RATE = lr;
}

void Backpropagation::setBatchSize(size_t bs)
{
	_batch_size = bs;
}

void Backpropagation::minimize()
{
	vector<const vector<double>*> batch_in(_batch_size);
	vector<const vector<double>*> batch_out(_batch_size);

	for (size_t i = 0; i < _batch_size; i++)
	{
		int z = rand() % _d->getIns(TRAIN).size();
		batch_in[i] = _d->getIns(TRAIN)[z];
		batch_out[i] = _d->getOuts(TRAIN)[z];
	}
	backpropagate(batch_in, batch_out);
}


vector<vector<vector<double>>> Backpropagation::getBackpropagationShifts(const vector<double>& in, const vector<double>& out)
{
	vector<vector<vector<double>>> dw(_n->getLayers().size());
	auto out_exp = _n->predict(in);
	for (int i = _n->getLayers().size() - 1; i >= 1; --i)
	{
		auto _dw = move(_n->getLayers()[i]->getBackpropagationShifts(out));
		dw[_n->getLayers()[i]->getId()] = _dw;
	}
	return move(dw);

}

void  Backpropagation::backpropagate(const vector<const vector<double>*>& ins, const vector<const vector<double>*>& outs)
{
	vector<vector<vector<double>>> dw(_n->getLayers().size());
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
					dw[j][k].resize(_dw[j][k].size(), 0);
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

	_n->shiftBackWeights(dw);
}

