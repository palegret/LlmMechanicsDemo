

using System.Collections.Generic;

namespace LlmMechanicsDemo.Library;

public class DeepNeuralNetwork
{
    public List<NeuronLayer> Layers { get; private set; }

    public DeepNeuralNetwork(List<NetworkHyperparameters> layerConfigs)
    {
        Layers = new List<NeuronLayer>();
        for (int i = 0; i < layerConfigs.Count; i++)
        {
            Layers.Add(new NeuronLayer(layerConfigs[i]));
        }
    }

    public double[] Process(double[] inputs)
    {
        double[] currentSignal = inputs;

        for (int i = 0; i < Layers.Count; i++)
        {
            currentSignal = Layers[i].ProcessLayer(currentSignal);
        }

        return currentSignal;
    }
}