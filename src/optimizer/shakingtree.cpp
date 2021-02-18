#include "shakingtree.h"
#include <algorithm>
#include <random>
#include <chrono>



Shakingtree::Shakingtree()
{

}


Shakingtree::~Shakingtree()
{
}



void Shakingtree::minimize()
{
	minimizeComplex();
}


void Shakingtree::minimizeBasic()
{
	mapParameters();

	//get a score
	int batch_size = 20;
	int weight_amplitude = 5;
	double s = getScore(TRAIN, batch_size);

	//choose a parameter to change
	int i = rand() % _p.size();
	double oldp = _p[i]->weight();
	_p[i]->alterWeight(random(-weight_amplitude, weight_amplitude));

	//evaluate the new score
	double news = getScore(TRAIN, batch_size);

	//if the new score (loss) is bigger, we keep the old weight
	if (news > s)
		_p[i]->alterWeight(oldp);
	return;
}

void Shakingtree::minimizeBasicLarger()
{
	mapParameters();

	//get a score
	int batch_size = 100;
	int weight_amplitude = 5;
	size_t n_new_parameters = 5;// int(0.1 * _p_ids.size());
	unsigned seed = (unsigned int)(std::chrono::system_clock::now().time_since_epoch().count());
	std::shuffle(_p_ids.begin(), _p_ids.end(), std::default_random_engine(seed));
	double s = getScore(TRAIN, batch_size);

	//choose multiple parameters to change
	vector<double> old_p;
	for (size_t j = 0; j < n_new_parameters; j++)
	{
		old_p.push_back(_p[_p_ids[j]]->weight());
		_p[_p_ids[j]]->alterWeight(random(-weight_amplitude, weight_amplitude));
	}

	//evaluate the new score
	double new_s = getScore(TRAIN, batch_size);

	//if the new score (loss) is bigger, we keep the old weight
	if (new_s > s)
		for (size_t j = 0; j < n_new_parameters; j++)
			_p[_p_ids[j]]->alterWeight(old_p[j]);
	return;
}


void Shakingtree::minimizeComplex()
{
	mapParameters();
	
	//PHASE 1 We compute the previous score
	size_t EVALSIZE = 100;
	srand(_total_iter);
	double score = getScore(TRAIN, EVALSIZE);


	//We apply the shift to the weights
	std::normal_distribution<double> rnorm(0, _step);
	vector<double> neww;
	neww.resize(_p.size());
	for (size_t i = 0; i < _p.size(); i++)
	{
		neww[i] = rnorm(_generator);

		//apply the delta
		_p[i]->shiftWeight(neww[i]);
	}

	//PHASE 2 : We compute the delta
	//evaluate score
	srand(_total_iter);
	double delta_score = getScore(TRAIN, EVALSIZE) - score;

	_delta_score.push_back(delta_score);
	_shift.push_back(neww);


	//PHASE 2B : We remove the shift
	for (size_t i = 0; i < _p.size(); i++)
	{
		//remove the delta
		_p[i]->resetLastShift();
	}


	//PHASE 3 : With a large enough memory, we hope to be able to analyze the delta score and the shift such that we know what a good shift is
	if (_shift.size() == _itmod)
	{

		//apply the averaged shift
		uint gscore = 0;
		for (size_t j = 0; j < _shift.size(); j++)
			if (_delta_score[j] < 0) // || _total_iter % 10 == 0)
			{
				for (size_t i = 0; i < _p.size(); i++)
				{
					double wsum = 0;
					wsum += _shift[j][i];
					_p[i]->shiftWeight(wsum*LEARNING_RATE);
				}
				gscore++;
			}
		
		//are we getting better? if no, other strategies may be applied
		if (gscore == 0)
			_nogoodscore_iter++;
		else
			_nogoodscore_iter = 0;

		//clear shift, delta score and increment iter
		_shift.clear();
		_delta_score.clear();
		_total_iter++;
	}


	return;
}


void Shakingtree::minimizeBasicPerLayer()
{
	mapParameters();

	
	vector<Edge*>& layer = _p2[rand()%_p2.size()];

	for (size_t i = 0; i < 1000; i++)
	{
		double s = getScore(TRAIN, 100);
		double neww = random(-7, 7);
		int i_edge = rand() % layer.size();
		double oldw = layer[i_edge]->weight();
		layer[i_edge]->alterWeight(neww);
		double news = getScore(TRAIN, 100);
		if (news > s)
			layer[i_edge]->shiftWeight(oldw);
	}

	return;
}


// easier access to parameters for the optimizer
void Shakingtree::mapParameters()
{
	if (_p.size() == 0)
	{
		auto& w = _n->getEdges();
		for (size_t i = 0; i < w.size(); i++)
			for (size_t j = 0; j < w[i].size(); j++)
			{
				for (size_t k = 0; k < w[i][j].size(); k++)
					_p.push_back(w[i][j][k]);
				_p2.push_back(move(w[i][j]));
			}
		cout << _p.size() << " mapped parameters" << endl;
	}
	for (size_t i = 0; i < _p.size(); i++)  _p_ids.push_back(i);
}







