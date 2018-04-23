// NeuralNetTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

using namespace ZahnNN;

void printArr(std::vector<double>& arr){
	std::cout << "[";
	for (unsigned k = 0; k < arr.size(); k++){
		std::cout << arr[k];//to_n_decimals(arr[k], 3);

		if (k < arr.size() - 1){
			std::cout << ", ";
		}
	}
	std::cout << "]" << std::endl;
}

int main()
{
	srand(timeGetTime());

	std::vector<unsigned> topo;

	topo.push_back(2);
	topo.push_back(4);
	topo.push_back(1);

	Neuron::DefaultActivationFunction = &Neuron::ActivationFastSigmoid;
	Neuron::DefaultActivationFunctionDerivative = &Neuron::ActivationFastSigmoidDerivative;

	NeuralNet net(topo);

	auto& outputLayer = net.getLayer(net.getLayerCount() - 1);

	TrainingData t_data[4];

	t_data[0].input = { 0.0, 0.0 };
	t_data[0].target = { 0.0 };

	t_data[1].input = { 1.0, 0.0 };
	t_data[1].target = { 1.0 };

	t_data[2].input = { 0.0, 1.0 };
	t_data[2].target = { 1.0 };

	t_data[3].input = { 1.0, 1.0 };
	t_data[3].target = { 0.0 };
	
	for (unsigned i = 0; i < 1000000; i++){
		unsigned r = rand() % 4;

		net.train(t_data[r].input, t_data[r].target);
	}

	for (unsigned i = 0; i < 4; i++){
		std::cout << "Input: ";
		printArr(t_data[i].input);

		std::cout << "Target: ";
		printArr(t_data[i].target);

		std::cout << "Prediction: ";
		net.feedForward(t_data[i].input);
		auto prediction = net.predict();
		printArr(prediction);

		std::cout << std::endl;
	}

	std::cin.ignore();

	return 0;
}

