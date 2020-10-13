#ifndef FUNCTIONS_H
#define FUNCTIONS_H


#include <iostream>
#include <vector>
#include <utility>
#include <limits>
#include <string>

using namespace std;

//Sigmoid Function
double sigmoid(double x);
double sigmoid_derivative(double x);

//Relu Function
double relu(double x);
double relu_derivative(double x);

//Random float getter function
double random(double low,double high);

double distanceVector(const vector<double>& v1, const vector<double>& v2);


#endif // FUNCTIONS_H
