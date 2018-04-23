#include "pch.h"
#include "Neuron.h"

namespace ZahnNN{

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


	double(*Neuron::DefaultActivationFunction)(double, void*, unsigned) = &Neuron::ActivationSigmoid;
	double(*Neuron::DefaultActivationFunctionDerivative)(double, void*, unsigned) = &Neuron::ActivationSigmoidDerivative;

	void* Neuron::DefaultActivationFunctionArgs = {};
	unsigned Neuron::DefaultActivationFunctionArgc = 0;

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

	double Neuron::getDerivitiveResult(){
		return m_activationFunctionDerivative(getInputVal(), m_activationArgs, m_activationArgc);
	}

	double Neuron::getInputValNoWeight(){
		double result = 0.0;

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			if ((*m_previousLayer)[i].m_isActive == false)
				continue;

			result += (*m_previousLayer)[i].m_outputVal;
		}

		return result;
	}

	double Neuron::getInputVal(){
		double result = 0.0;

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			Neuron& neuron = (*m_previousLayer)[i];

			if (neuron.m_isActive == false) continue;

			result += neuron.getOutputVal() * neuron.getOutputWeights()[m_index].weight;
		}

		return result;
	}

	std::vector<double> Neuron::getWeightedErrors(){
		std::vector<double> result;

		double sumWeight = 0.0;

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			sumWeight += abs((*m_previousLayer)[i].getOutputWeights()[m_index].weight);
		}

		for (unsigned i = 0; i < m_previousLayer->size(); i++){
			result.push_back(
				(*m_previousLayer)[i].getOutputWeights()[m_index].weight / sumWeight * m_error
			);
		}

		return result;
	}

	void Neuron::feedForward(){
		double input = getInputVal();

		m_outputVal = m_activationFunction(input, m_activationArgs, m_activationArgc);
	}

	void Neuron::calculateInputDeltaWeights(double eta, double alpha){
		/*std::cout << "Delta Weights: ";
		std::cout << "[";*/
		for (unsigned n = 0; n < m_previousLayer->size(); n++){
			Neuron& neuron = (*m_previousLayer)[n];
			if (neuron.m_isActive == false)
				continue;

			double old_delta_w = neuron.m_outputWeights[m_index].deltaWeight;

			double delta_w = eta * m_error * getDerivitiveResult() * neuron.getOutputVal()
				+ alpha * old_delta_w;

			neuron.m_outputWeights[m_index].deltaWeight = delta_w;

			//std::cout << (to_n_decimals(delta_w, 3)) << ", ";
		}
		//std::cout << "]" << std::endl;
	}

	void Neuron::updateOutputWeights(){
		for (auto& connection : m_outputWeights){
			connection.weight += connection.deltaWeight;
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

		return x / (1.0 + abs(x));
	}
#
	double Neuron::ActivationFastSigmoidDerivative(double x, void* args, unsigned argc)
	{
		assert(argc == 0);

		//return (1.0 / (abs(x) + 1.0)) - (pow(x, 2) / (abs(x) * pow(abs(x) + 1.0, 2)));
		return 1.0 / pow(abs(x) + 1.0, 2);
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