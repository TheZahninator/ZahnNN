#include "pch.h"
#include "NeuralNet.h"

namespace ZahnAI{

	NeuralNet::NeuralNet(const std::vector<unsigned int> &topology)
	{
		for (unsigned int i = 0; i < topology.size(); i++){
			m_layers.push_back(Layer());

			bool isLastLayer = i == topology.size() - 1;

			for (unsigned int j = 0; j <= topology[i]; j++){
				unsigned numOutputs = isLastLayer ? 0 : topology[i + 1];
				Neuron newNeuron(numOutputs, j);
				m_layers[i].push_back(newNeuron);
				//std::cout << "new neuron" << std::endl;


			}
			m_layers[i].back().setOutputVal(1.0);
			m_layers[i].back().setChanceToActivate(1.0);
		}
		for (unsigned i = 0; i < m_layers.front().size(); i++)
			m_layers.front()[i].setChanceToActivate(1.0);

		for (unsigned i = 0; i < m_layers.back().size(); i++)
			m_layers.back()[i].setChanceToActivate(1.0);

		//std::cout << "New neural network with size: ";
		//for (unsigned i : topology){
		//	std::cout << i << " ";
		//}
		//std::cout << std::endl << std::endl;

		m_isTraining = false;
	}


	NeuralNet::~NeuralNet()
	{
	}

	void NeuralNet::feedForward(const std::vector<double> &inputVals){
		assert(inputVals.size() == m_layers[0].size() - 1);

		//Deactivate random neurons in the hidden layers
		for (unsigned i = 1; i < m_layers.size() - 1; i++){
			for (unsigned n = 0; n < m_layers[i].size(); n++){
				m_layers[i][n].activate(m_isTraining);
			}
		}

		//Feed input into input layer
		for (unsigned i = 0; i < inputVals.size(); i++){
			m_layers[0][i].setOutputVal(inputVals[i]);
		}


		for (unsigned i = 1; i < m_layers.size(); i++){

			Layer &prevLayer = m_layers[i - 1];

			for (unsigned n = 0; n < m_layers[i].size() - 1; n++){
				if (m_layers[i][n].getActive() == false)
					continue;

				m_layers[i][n].feedForward(prevLayer);
			}
		}
	}

	void NeuralNet::backProp(const std::vector<double> &targetVals){
		//Calculate net error
		Layer &outputLayer = m_layers.back();
		m_error = 0.0;

		for (unsigned n = 0; n < outputLayer.size() - 1; n++){
			double delta = targetVals[n] - outputLayer[n].getOutputVal();
			m_error += delta * delta;
		}
		m_error /= outputLayer.size() - 1; //average error squared
		m_error = sqrt(m_error); //RMS

		//Calculate output layer gradients
		for (unsigned n = 0; n < outputLayer.size() - 1; n++){
			outputLayer[n].calcOutputGradients(targetVals[n]);
		}

		//Calculate gradients on hidden layers
		for (int layer = m_layers.size() - 2; layer >= 0; layer--){
			Layer &hiddenLayer = m_layers[layer];
			Layer &nextLayer = m_layers[layer + 1];

			for (unsigned n = 0; n < hiddenLayer.size(); n++){
				if (hiddenLayer[n].getActive() == false)
					continue;

				hiddenLayer[n].calcHiddenGradients(nextLayer);
			}
		}

		//For all layers from outputs to first hidden layer, update connection weights
		for (unsigned i = m_layers.size() - 1; i > 0; i--){
			Layer &layer = m_layers[i];
			Layer &prevLayer = m_layers[i - 1];

			for (unsigned n = 0; n < layer.size() - 1; n++){
				if (layer[n].getActive() == false)
					continue;

				layer[n].updateInputWeights(prevLayer);
			}
		}
	}

	void NeuralNet::getResults(std::vector<double> &resultVals){
		resultVals.clear();

		Layer &currentLayer = m_layers.back();
		for (unsigned i = 0; i < currentLayer.size() - 1; i++){
			resultVals.push_back(currentLayer[i].getOutputVal());
			//std::cout << currentLayer[i].getOutputVal() << std::endl;
		}
	}

}