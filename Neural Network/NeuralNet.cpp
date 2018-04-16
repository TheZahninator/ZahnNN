#include "pch.h"
#include "NeuralNet.h"

namespace ZahnNN{

	double NeuralNet::Default_alpha = 0.1;

	NeuralNet::NeuralNet(const std::vector<unsigned int> &topology)
	{
		for (unsigned int i = 0; i < topology.size(); i++){
			m_layers.push_back(Layer());

			bool isLastLayer = i == topology.size() - 1;

			for (unsigned int j = 0; j <= topology[i] - (int)(isLastLayer); j++){
				unsigned numOutputs = isLastLayer ? 0 : topology[i + 1];
				Neuron newNeuron(numOutputs, j);

				m_layers[i].push_back(newNeuron);
			}

			m_layers[i].back().setOutputVal(1.0);
			m_layers[i].back().setChanceToActivate(1.0);
		}
		for (unsigned i = 0; i < m_layers.front().size(); i++)
			m_layers.front()[i].setChanceToActivate(1.0);

		for (unsigned i = 0; i < m_layers.back().size(); i++)
			m_layers.back()[i].setChanceToActivate(1.0);

		for (unsigned i = 0; i < m_layers.size(); i++){
			for (auto& neuron : m_layers[i]){
				if (i != 0){
					neuron.setPrevLayer(&m_layers[i - 1]);
				}
				if (i < m_layers.size() - 1){
					neuron.setNextLayer(&m_layers[i + 1]);
				}
			}
		}

		m_isTraining = false;

		m_alpha = Default_alpha;
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
			bool isOutputLayer = i == m_layers.size() - 1;

			Layer &prevLayer = m_layers[i - 1];

			for (unsigned n = 0; n < m_layers[i].size() - int(!isOutputLayer); n++){
				if (m_layers[i][n].isActive() == false)
					continue;

				m_layers[i][n].feedForward();
			}
		}
	}

	void NeuralNet::backProp(const std::vector<double> &targetVals){
		auto &outputLayer = m_layers.back();
		auto outputVals = predict();

		assert(targetVals.size() == outputLayer.size());
		
		std::vector<double> errors;

		for (unsigned i = 0; i < targetVals.size(); i++){
			errors.push_back(targetVals[i] - outputVals[i]);
		}

		for (unsigned i = m_layers.size() - 1; i > 0; i--){
			bool isOutputLayer = i == m_layers.size() - 1;

			auto& currentLayer = m_layers[i];

			//Get errors
			for (unsigned j = 0; j < currentLayer.size(); j++){
				currentLayer[j].setError(errors[j]);
			}

			errors.clear();

			for (unsigned j = 0; j < currentLayer.size() - (int)(!isOutputLayer); j++){
				auto& neuron = currentLayer[j];
				
				for (unsigned k = 0; k < neuron.getWeightedErrors().size(); k++){
					double err = neuron.getWeightedErrors()[k];
					
					if (j == 0){
						errors.push_back(err);
					}
					else{
						errors[k] += err;
					}
				}
			}
		}

		//Adjust weights
		for (unsigned i = 1; i < m_layers.size(); i++){
			bool isOutputLayer = i == m_layers.size() - 1;

			//std::cout << "Layer [" << i << "]" << std::endl;
			for (unsigned n = 0; n < m_layers[i].size() - (int)(!isOutputLayer); n++){
				//std::cout << "Neuron [" << n << "] ";
				m_layers[i][n].calculateInputDeltaWeights(m_alpha);
			}
			//std::cout << std::endl;
		}

		for (Layer& layer : m_layers){
			for (Neuron& neuron : layer){
				neuron.updateOutputWeights();
			}
		}
	}

	void NeuralNet::train(const std::vector<double> &inputVals, const std::vector<double> &targetVals){
		feedForward(inputVals);
		backProp(targetVals);
	}

	void NeuralNet::getResults(std::vector<double> &resultVals){
		resultVals.clear();

		Layer &outputLayer = m_layers.back();
		for (unsigned i = 0; i < outputLayer.size(); i++){
			resultVals.push_back(outputLayer[i].getOutputVal());
		}
	}

	const std::vector<double> NeuralNet::predict(){
		std::vector<double> results;

		getResults(results);
		
		return results;
	}

	void NeuralNet::mutate(double mutationRate){
		for (Layer& layer : m_layers){
			for (Neuron& neuron : layer){
				for (Connection& conn : neuron.getOutputWeights()){
					double delta = map((double)rand(), 0.0, (double)RAND_MAX, 0.0, mutationRate);
					delta *= rand() % 2 == 0 ? -1 : 1;
					conn.weight += delta;
				}
			}
		}
	}

	NeuralNet* NeuralNet::clone(){
		std::vector<unsigned> topo;

		for (unsigned i = 0; i < m_layers.size(); i++){
			topo.push_back(m_layers[i].size() - 1);
		}

		NeuralNet* child = new NeuralNet(topo);

		for (unsigned i = 0; i < m_layers.size(); i++){
			for (unsigned j = 0; j < m_layers[i].size(); j++){
				child->m_layers[i][j].setOutputWeights(m_layers[i][j].getOutputWeights());
			}
		}

		child->setAlpha(m_alpha);

		return child;
	}

	NeuralNet* NeuralNet::crossover(NeuralNet* partner){
		std::vector<unsigned> topo;

		for (unsigned i = 0; i < m_layers.size(); i++){
			topo.push_back(m_layers[i].size() - 1);
		}

		NeuralNet* child = new NeuralNet(topo);
		child->setAlpha(m_alpha);

		for (unsigned i = 0; i < m_layers.size(); i++){
			for (unsigned j = 0; j < m_layers[i].size(); j++){
				for (unsigned k = 0; k < m_layers[i][j].getOutputWeights().size(); k++){

					unsigned num = rand() % 2;
					if (num == 0){
						child->m_layers[i][j].getOutputWeights()[k].weight = m_layers[i][j].getOutputWeights()[k].weight;
					}
					else{
						child->m_layers[i][j].getOutputWeights()[k].weight = partner->m_layers[i][j].getOutputWeights()[k].weight;
					}

				}
			}
		}

		return child;
	}
}