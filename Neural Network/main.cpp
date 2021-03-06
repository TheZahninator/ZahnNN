#include "pch.h"

#include "NeuralNet.h"
#include <iostream>

using namespace ZahnAI;



int main(){
	srand((unsigned int) time(nullptr));

#if 0
	std::vector<unsigned int> topo({ 3, unsigned(rand() % 20 + 1), unsigned(rand() % 20 + 1), 1 });
#else
	std::vector<unsigned int> topo({ 3, 8, 8, 1 });
#endif
	NeuralNet net(topo);

	Neuron::setAlpha(0.25);
	Neuron::setETA(0.3);
	unsigned numTrainingSessions = 1000;

	//Training data for an or gate
	std::vector <TrainingData> trainingData;
	{
		TrainingData td;
		td.input.push_back(0.0);
		td.input.push_back(0.0);
		td.input.push_back(0.0);
		td.target.push_back(0.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(0.0);
		td.input.push_back(1.0);
		td.input.push_back(0.0);
		td.target.push_back(1.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(0.0);
		td.input.push_back(0.0);
		td.input.push_back(1.0);
		td.target.push_back(1.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(0.0);
		td.input.push_back(1.0);
		td.input.push_back(1.0);
		td.target.push_back(1.0);
		trainingData.push_back(td);
	}


	//Training data for an xor gate
	{
		TrainingData td;
		td.input.push_back(1.0);
		td.input.push_back(0.0);
		td.input.push_back(0.0);
		td.target.push_back(0.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(1.0);
		td.input.push_back(1.0);
		td.input.push_back(0.0);
		td.target.push_back(1.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(1.0);
		td.input.push_back(0.0);
		td.input.push_back(1.0);
		td.target.push_back(1.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(1.0);
		td.input.push_back(1.0);
		td.input.push_back(1.0);
		td.target.push_back(0.0);
		trainingData.push_back(td);
	}


	//Training data for an and gate
	{
		TrainingData td;
		td.input.push_back(0.5);
		td.input.push_back(0.0);
		td.input.push_back(0.0);
		td.target.push_back(0.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(0.5);
		td.input.push_back(1.0);
		td.input.push_back(0.0);
		td.target.push_back(0.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(0.5);
		td.input.push_back(0.0);
		td.input.push_back(1.0);
		td.target.push_back(0.0);
		trainingData.push_back(td);
	}
	{
		TrainingData td;
		td.input.push_back(0.5);
		td.input.push_back(1.0);
		td.input.push_back(1.0);
		td.target.push_back(1.0);
		trainingData.push_back(td);
	}


	net.setTrainingMode(false);

	for (unsigned i = 0; i < numTrainingSessions; i++){
		//std::cout << std::endl << "Pass: " << i << std::endl << std::endl;
		for (TrainingData td : trainingData){
			
			net.feedForward(td.input);
			//std::cout << "Feeding: ";
			//for (double d : td.input)
			//	std::cout << d << " ";
			//std::cout << std::endl;

			std::vector<double> result;
			net.getResults(result);
			//std::cout << "Result: " << round(result[0]) << std::endl;
			//std::cout << "Target: " << td.target[0] << std::endl;

			net.backProp(td.target);

			//std::cout << std::endl;
		}
	}

	//test
	net.setTrainingMode(false);

	for (TrainingData td : trainingData){

		net.feedForward(td.input);
		std::cout << "Feeding: ";
		for (double d : td.input)
			std::cout << d << " ";
		std::cout << std::endl;

		std::vector<double> result;
		net.getResults(result);
		std::cout << "Result: " << round(result[0]) << std::endl;
		std::cout << "Target: " << td.target[0] << std::endl;

		std::cout << std::endl;
	}
	
	system("PAUSE");
	return 0;
}