#include "pch.h"
#include "Neuron.h"

namespace ZahnAI{

	double Neuron::DefaultStepThreshold = 0.5;

	double(*Neuron::ActivationFunctions[Neuron::NumActivationFunctions])(double, void*, unsigned) = {
		Neuron::ActivationStep,
		Neuron::ActivationTanH,
		Neuron::ActivationSigmoid,
		Neuron::ActivationFastSigmoid,
		Neuron::ActivationReLu
	};

	double(*Neuron::ActivationFunctionDerivatives[Neuron::NumActivationFunctions])(double, void*, unsigned) = {
		Neuron::ActivationSigmoidDerivative,
		Neuron::ActivationTanHDerivative,
		Neuron::ActivationSigmoidDerivative,
		Neuron::ActivationFastSigmoidDerivative,
		Neuron::ActivationReLuDerivative
	};

	void* Neuron::ActivationFunctionsArgs[Neuron::NumActivationFunctions] = {
		{(void*)&Neuron::DefaultStepThreshold},
		{},
		{},
		{},
		{}
	};
	
	unsigned Neuron::ActivationFunctionsArgc[Neuron::NumActivationFunctions] = {
		1,
		0,
		0,
		0,
		0
	};


	double(*Neuron::DefaultActivationFunction)(double, void*, unsigned) = &Neuron::ActivationFastSigmoid;
	double(*Neuron::DefaultActivationFunctionDerivative)(double, void*, unsigned) = &Neuron::ActivationFastSigmoidDerivative;

	void* Neuron::DefaultActivationFunctionArgs = {};
	unsigned Neuron::DefaultActivationFunctionArgc = 0;


	double Neuron::eta = 0.01; //Learning rate
	double Neuron::alpha = 0.025;	//Momentum

	bool Neuron::isTraining = false;

	Neuron::Neuron(unsigned outputs, unsigned index, bool randomActivationFunc)
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

		if (randomActivationFunc){
			unsigned i = rand() % NumActivationFunctions;
			setActivationFunction(ActivationFunctions[i], ActivationFunctionDerivatives[i], ActivationFunctionsArgs[i], ActivationFunctionsArgc[i]);
		}
		else{
			setActivationFunction(DefaultActivationFunction, DefaultActivationFunctionDerivative, DefaultActivationFunctionArgs, DefaultActivationFunctionArgc);
		}
	}


	Neuron::~Neuron()
	{
	}

	std::vector<double> Neuron::getWeightedErrors(){
		std::vector<double> result;

		double sumWeight = 0.0;

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			sumWeight += (*m_previousLayer.get())[i].getOutputWeights()[m_index].weight;
		}

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			result.push_back(
				(*m_previousLayer.get())[i].getOutputWeights()[m_index].weight / sumWeight * m_error
			);
		}

		return result;
	}

	void Neuron::feedForward(){
		double sum = 0.0;

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			if ((*m_previousLayer.get())[i].m_isActive == false)
				continue;

			sum += (*m_previousLayer.get())[i].m_outputVal *
				(*m_previousLayer.get())[i].m_outputWeights[m_index].weight;
		}

		m_outputVal = m_activationFunction(sum, m_activationArgs, m_activationArgc);
	}

	double Neuron::transferFunction(double x){
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
		m_gradient = delta * m_activationFunctionDerivative(m_outputVal, m_activationArgs, m_activationArgc);
	}

	void Neuron::calcHiddenGradients(Layer& nextLayer){
		double dow = sumDOW(nextLayer);
		m_gradient = dow * m_activationFunctionDerivative(m_outputVal, m_activationArgs, m_activationArgc);
	}

	void Neuron::updateInputWeights(Layer& prevLayer, double eta, double alpha){
		for (unsigned n = 0; n < prevLayer.size(); n++){
			Neuron& neuron = prevLayer[n];
			if (neuron.m_isActive == false)
				continue;

			double oldDeltaWeight = neuron.m_outputWeights[m_index].deltaWeight;

			double newDeltaWeight = eta * neuron.getOutputVal() * m_gradient + alpha * oldDeltaWeight;

			neuron.m_outputWeights[m_index].deltaWeight = newDeltaWeight;
			neuron.m_outputWeights[m_index].weight += newDeltaWeight;
		}
	}


	//Activation functions
	void Neuron::setActivationFunction(
		double(*func)(double, void*, unsigned), 
		double(*funcDerivative)(double, void*, unsigned), 
		void* args, 
		unsigned argc
	)
	{
		m_activationFunction = func;
		m_activationFunctionDerivative = funcDerivative;
		m_activationArgs = args;
		m_activationArgc = argc;
	}


	double Neuron::ActivationStep(double x, void* args, unsigned argc)
	{
		assert(argc == 1);

		double threshold = ((double*)args)[0];

		return x > threshold ? 1.0 : 0.0;
	}

	double Neuron::ActivationStepDerivative(double x, void* args, unsigned argc)
	{
		assert(argc == 1);

		double threshold = ((double*)args)[0];

		return x == threshold ? 1.0 : 0.0;
	}

	double Neuron::ActivationTanH(double x, void* args, unsigned argc)
	{
		assert(argc == 0);

		return tanh(x);
	}

	double Neuron::ActivationTanHDerivative(double x, void* args, unsigned argc)
	{
		assert(argc == 0);

		return 1.0 - tanh(x) * tanh(x);
	}

	double Neuron::ActivationSigmoid(double x, void* args, unsigned argc)
	{
		assert(argc == 0);

		return 1.0 / (1.0 + exp(-x));
	}

	double Neuron::ActivationSigmoidDerivative(double x, void* args, unsigned argc)
	{
		assert(argc == 0);

		return ActivationSigmoid(x, args, argc) * (1.0 - ActivationSigmoid(x, args, argc));
	}

	double Neuron::ActivationFastSigmoid(double x, void* args, unsigned argc)
	{
		assert(argc == 0);

		return abs(x) / (1.0 + abs(x));
	}
#
	double Neuron::ActivationFastSigmoidDerivative(double x, void* args, unsigned argc)
	{
		return ActivationSigmoidDerivative(x, args, argc);
	}

	double Neuron::ActivationReLu(double x, void* args, unsigned argc)
	{
		assert(argc == 1);

		double factor = ((double*)args)[0];

		return std::max(factor * x, x);
	}

	double Neuron::ActivationReLuDerivative(double x, void* args, unsigned argc)
	{
		assert(argc == 1);

		double factor = ((double*)args)[0];

		return x >= 0.0 ? 1.0 : factor;
	}
}