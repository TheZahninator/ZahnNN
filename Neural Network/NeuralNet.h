#ifndef ZAHNNN_NEURAL_NET_H
#define ZAHNNN_NEURAL_NET_H

#include "Neuron.h"
#include "Math.h"

namespace ZahnNN{

	class NeuralNet
	{
	public:
		static double Default_alpha;
		static double Default_eta;

		NeuralNet(const std::vector<unsigned int> &topology);
		~NeuralNet();

		void feedForward(const std::vector<double> &inputVals);
		void backProp(const std::vector<double> &targetVals);
		void getResults(std::vector<double> &resultVals);

		void train(const std::vector<double> &inputVals, const std::vector<double> &targetVals);
		const std::vector<double> predict();

		void setTrainingMode(bool b){ m_isTraining = b; Neuron::setTraining(b); }

		void setAlpha(double alpha){ m_alpha = alpha; }
		void setEta(double eta){ m_eta = eta; }

		void mutate(double mutationRate);

		NeuralNet* clone();
		NeuralNet* crossover(NeuralNet* partner);

		unsigned getLayerCount(){ return m_layers.size(); }
		Layer& getLayer(unsigned x){ return m_layers[x]; }

		const std::vector<unsigned>& getTopology() { return m_topology; }

	private:
		bool m_isTraining;

		std::vector<unsigned> m_topology;

		std::vector<Layer> m_layers;

		double m_alpha;
		double m_eta;
	};

}

struct TrainingData{
	std::vector<double> input;
	std::vector<double> target;
};

#endif