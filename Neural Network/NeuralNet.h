#pragma once

#include "Neuron.h"
#include <iostream>

//typedef std::vector<Neuron> Layer;

class NeuralNet
{
public:
	NeuralNet(const std::vector<unsigned int> &topology);
	~NeuralNet();

	void feedForward(const std::vector<double> &inputVals);
	void backProp(const std::vector<double> &targetVals);
	void getResults(std::vector<double> &resultVals);

	void setTrainingMode(bool b){ m_isTraining = b; Neuron::setTraining(b); }

private:
	bool m_isTraining;
	double m_error;

	std::vector<Layer> m_layers;
};

