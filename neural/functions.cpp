#include "functions.h"

//Sigmoid Function
double sigmoid(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

double sigmoid_derivative(double x)
{
	return sigmoid(x) * (1 - sigmoid(x));
}

//Relu Function
double relu(double x)
{
	if (x > 0)
		return x;
	return 0;
}
double relu_derivative(double x)
{
	if (x > 0)
		return 1;
	return 0;
}

//Random float getter function
double random(double low, double high)
{
    return low + static_cast <double> (rand()) /( static_cast <double> (RAND_MAX/(high-low)));
}




