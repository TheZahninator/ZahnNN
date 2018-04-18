#pragma once

namespace ZahnNN{

	class Neuron;

	typedef std::vector<Neuron> Layer;

	struct Connection{
		double weight;
		double deltaWeight;
	};

	class Neuron
	{
	public:
		Neuron(unsigned, unsigned, bool = false);
		~Neuron();

		static void setTraining(bool b){ isTraining = b; }

		void setChanceToActivate(double d){ m_chanceToActivate = d; }

		void activate(bool randomize){
			if (randomize){
				double d = rand() / double(RAND_MAX);
				m_isActive = d <= m_chanceToActivate;
			}
			else{
				m_isActive = true;
			}
		}

		bool isActive(){ return m_isActive; }

		void setOutputVal(double val){ m_outputVal = val; }
		double getOutputVal(){
			if (isTraining)
				return m_outputVal;

			return m_outputVal * m_chanceToActivate;
		}

		double getInputValNoWeight();
		double getInputVal();

		double getDerivitiveResult();

		void feedForward();

		void setError(double x){ m_error = x; }
		double getError(){ return m_error; }

		std::vector<double> getWeightedErrors();

		void calculateInputDeltaWeights(double eta, double alpha);
		void updateOutputWeights();

		std::vector<Connection>& getOutputWeights(){ return m_outputWeights; }
		void setOutputWeights(const std::vector<Connection>& weights){ m_outputWeights = weights; }

		void setPrevLayer(Layer* prev){ m_previousLayer.reset(prev); }
		void setNextLayer(Layer* next){ m_nextLayer.reset(next); }

	private:
		bool m_isActive;
		double m_chanceToActivate;
		static bool isTraining;

		unsigned m_index;

		double m_error;

		double m_outputVal;
		std::vector<Connection> m_outputWeights;

		std::shared_ptr<Layer> m_previousLayer;
		std::shared_ptr<Layer> m_nextLayer;

		static double randomWeight(){ return rand() / double(RAND_MAX) - 0.5; }

		double(*m_activationFunction)(double, void*, unsigned);
		double(*m_activationFunctionDerivative)(double, void*, unsigned);

		void* m_activationArgs;
		unsigned m_activationArgc;

	public:
		void setActivationFunction(double(*func)(double, void*, unsigned), double(*funcDerivative)(double, void*, unsigned), void* args, unsigned argc);

		static double DefaultStepThreshold;

		//Activation functions
		static double ActivationStep(double, void*, unsigned);
		static double ActivationStepDerivative(double, void*, unsigned);

		static double ActivationTanH(double, void*, unsigned);
		static double ActivationTanHDerivative(double, void*, unsigned);

		static double ActivationSigmoid(double, void*, unsigned);
		static double ActivationSigmoidDerivative(double, void*, unsigned);

		static double ActivationFastSigmoid(double, void*, unsigned);
		static double ActivationFastSigmoidDerivative(double, void*, unsigned);

		static double ActivationReLu(double, void*, unsigned);
		static double ActivationReLuDerivative(double, void*, unsigned);

		//List of functions
		const static unsigned NumActivationFunctions = 5;
		
		static double(*ActivationFunctions[NumActivationFunctions])(double, void*, unsigned);
		static double(*ActivationFunctionDerivatives[NumActivationFunctions])(double, void*, unsigned);
		
		static void* ActivationFunctionsArgs[NumActivationFunctions];
		static unsigned ActivationFunctionsArgc[NumActivationFunctions];

		static double(*DefaultActivationFunction)(double, void*, unsigned);
		static double(*DefaultActivationFunctionDerivative)(double, void*, unsigned);

		static void* DefaultActivationFunctionArgs;
		static unsigned DefaultActivationFunctionArgc;
	};
}
