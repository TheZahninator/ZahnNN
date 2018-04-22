#include "stdafx.h"
#include "CppUnitTest.h"

#include <functional>

#include "../Neural Network/ZahnNN.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace ZahnNNTest
{
	TEST_CLASS(NeuralNetworkTest)
	{
	private:
		std::unique_ptr<ZahnNN::NeuralNet> m_net;

	public:

		TEST_METHOD_INITIALIZE(init)
		{
			std::vector<unsigned> topo;

			topo.push_back(2);
			topo.push_back(3);
			topo.push_back(1);

			m_net.reset(new ZahnNN::NeuralNet(topo));
		}

		TEST_METHOD(it_can_be_deconstructed)
		{
			m_net.reset();
		}

		TEST_METHOD(it_throws_an_exception_if_the_number_of_inputs_does_not_match_the_number_of_input_neurons)
		{
			std::vector<double> inputs;
			inputs.push_back(1.0);

			bool exceptionThrown = false;

			try {
				m_net->feedForward(inputs);
			}
			catch (std::invalid_argument e) {
				exceptionThrown = true;
			}

			Assert::IsTrue(exceptionThrown);

			inputs.push_back(0.0);
			inputs.push_back(0.5);

			try {
				m_net->feedForward(inputs);
			}
			catch (std::invalid_argument e) {
				exceptionThrown = true;
			}

			Assert::IsTrue(exceptionThrown);
		}

		TEST_METHOD(it_throws_an_exception_if_the_number_of_targets_does_not_match_the_number_of_target_neurons)
		{
			std::vector<double> targets;

			bool exceptionThrown = false;

			try {
				m_net->backProp(targets);
			}
			catch (std::invalid_argument e) {
				exceptionThrown = true;
			}

			Assert::IsTrue(exceptionThrown);

			targets.push_back(0.0);
			targets.push_back(0.5);

			try {
				m_net->backProp(targets);
			}
			catch (std::invalid_argument e) {
				exceptionThrown = true;
			}

			Assert::IsTrue(exceptionThrown);
		}
	};
}