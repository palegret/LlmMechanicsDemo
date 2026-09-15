namespace LlmMechanicsDemo.Library;

public class NeuronLayer
{
    public List<Neuron> Neurons { get; private set; }

    public NeuronLayer(NetworkHyperparameters config)
    {
        Neurons = [];
        for (int i = 0; i < config.NeuronCount; i++)
        {
            Neurons.Add(new Neuron(config));
        }
    }

    public double[] ProcessLayer(double[] inputs)
    {
        double[] layerOutputs = new double[Neurons.Count];
        for (int i = 0; i < Neurons.Count; i++)
        {
            layerOutputs[i] = Neurons[i].Predict(inputs);
        }
        return layerOutputs;
    }
}
