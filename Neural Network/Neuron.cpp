#include "pch.h"
#include "Neuron.h"

double Neuron::eta = 0.01; //Learning rate
double Neuron::alpha = 0.025;	//Momentum

bool Neuron::isTraining = false;

Neuron::Neuron(unsigned outputs, unsigned index)
{
	for (unsigned c = 0; c < outputs; c++){
		m_outputWeights.push_back(Connection());
		m_outputWeights.back().weight = randomWeight();
		m_outputWeights.back().deltaWeight = 0.0;
	}

	m_outputVal = 0.0;
	m_index = index;
	m_isActive = true;
	m_chanceToActivate = 0.5;
}


Neuron::~Neuron()
{
}

void Neuron::feedForward(Layer& prevLayer){
	double sum = 0.0;

	for (unsigned i = 0; i < prevLayer.size(); i++){
		if (prevLayer[i].m_isActive == false)
			continue;

		sum += prevLayer[i].m_outputVal * 
			   prevLayer[i].m_outputWeights[m_index].weight;
	}

	m_outputVal = transferFunction(sum);
}


double Neuron::transferFunction(double x){
	//tanh [-1; 1]
	return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
	return 1.0 - x * x;
}

double Neuron::sumDOW(Layer& nextLayer){
	double sum = 0.0;

	for (unsigned n = 0; n < nextLayer.size() - 1; n++){
		if (nextLayer[n].m_isActive == false)
			continue;

		sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
	}

	return sum;
}

void Neuron::calcOutputGradients(double target){
	double delta = target - m_outputVal;
	m_gradient = delta * transferFunctionDerivative(m_outputVal);
}

void Neuron::calcHiddenGradients(Layer& nextLayer){
	double dow = sumDOW(nextLayer);
	m_gradient = dow * transferFunctionDerivative(m_outputVal);
}

void Neuron::updateInputWeights(Layer& prevLayer){
	for(unsigned n = 0; n < prevLayer.size(); n++){
		Neuron& neuron = prevLayer[n];
		if (neuron.m_isActive == false)
			continue;

		double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

		double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

		neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
		neuron.m_outputWeights[m_index].weight += newDeltaWeight;
	}
}