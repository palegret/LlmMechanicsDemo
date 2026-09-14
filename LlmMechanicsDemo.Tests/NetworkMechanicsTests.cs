using Xunit;
using LlmMechanicsDemo.Library;

namespace LlmMechanicsDemo.Tests;

public class NetworkMechanicsTests : IDisposable
{
    private readonly string _testFilePath = "test_model.bin";

    [Fact]
    public void DeepNetwork_PassesSignalThroughMultipleLayers()
    {
        var config1 = new NetworkHyperparameters { InputCount = 2, NeuronCount = 3, UseReluActivation = true };
        var config2 = new NetworkHyperparameters { InputCount = 3, NeuronCount = 1, UseReluActivation = true };

        var deepNetwork = new DeepNeuralNetwork(new List<NetworkHyperparameters> { config1, config2 });

        var layer1 = deepNetwork.Layers[0];
        layer1.Neurons[0].Weights[0] = 1.5; layer1.Neurons[0].Weights[1] = -2.0; layer1.Neurons[0].Bias = -1.0;
        layer1.Neurons[1].Weights[0] = -2.0; layer1.Neurons[1].Weights[1] = 1.5; layer1.Neurons[1].Bias = 0.0;
        layer1.Neurons[2].Weights[0] = -1.0; layer1.Neurons[2].Weights[1] = 2.0; layer1.Neurons[2].Bias = -2.0;

        var layer2 = deepNetwork.Layers[1];
        layer2.Neurons[0].Weights[0] = 2.0;  
        layer2.Neurons[0].Weights[1] = 1.0;  
        layer2.Neurons[0].Weights[2] = -3.0; 
        layer2.Neurons[0].Bias = -5.0;       

        double[] rawInputs = [8.0, 2.0];
        double[] finalResult = deepNetwork.Process(rawInputs);

        Assert.Single(finalResult);
        Assert.Equal(9.0, finalResult[0]);
    }

    [Fact]
    public void Neuron_LearnsToAdjustItsOwnWeights_ThroughPractice()
    {
        var config = new NetworkHyperparameters { InputCount = 2, UseReluActivation = false };
        var studentNeuron = new Neuron(config);
        
        double[] inputsA = [1.0, 0.0];
        double targetA = 10.0;
        
        double[] inputsB = [0.0, 1.0];
        double targetB = -5.0;

        Assert.Equal(0.0, studentNeuron.Predict(inputsA));

        for (int epoch = 0; epoch < 100; epoch++)
        {
            studentNeuron.Train(inputsA, targetA, 0.1);
            studentNeuron.Train(inputsB, targetB, 0.1);
        }

        Assert.Equal(10.0, Math.Round(studentNeuron.Predict(inputsA), 2));
        Assert.Equal(-5.0, Math.Round(studentNeuron.Predict(inputsB), 2));
    }

    [Fact]
    public void Neuron_CanSaveAndLoadState_BypassingNeedToRetrain()
    {
        var config = new NetworkHyperparameters { InputCount = 2, UseReluActivation = false };
        var trainerNeuron = new Neuron(config);
        double[] inputs = [1.0, 0.0];
        
        for (int i = 0; i < 100; i++)
        {
            trainerNeuron.Train(inputs, 10.0, 0.1);
        }

        trainerNeuron.Save(_testFilePath);

        var userNeuron = new Neuron(config);
        Assert.Equal(0.0, userNeuron.Predict(inputs));

        userNeuron.Load(_testFilePath);
        Assert.Equal(10.0, Math.Round(userNeuron.Predict(inputs), 2));
    }

    public void Dispose()
    {
        if (File.Exists(_testFilePath))
        {
            File.Delete(_testFilePath);
        }
    }
}