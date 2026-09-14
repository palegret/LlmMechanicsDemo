namespace LlmMechanicsDemo.Library;

public class Neuron
{
    public double[] Weights { get; set; }
    public double Bias { get; set; }
    
    private readonly NetworkHyperparameters _config;

    public Neuron(NetworkHyperparameters config)
    {
        _config = config;
        Weights = new double[config.InputCount];
        
        for (int i = 0; i < config.InputCount; i++)
        {
            Weights[i] = 0.0;
        }
        
        Bias = 0.0;
    }

    public double Predict(double[] inputs)
    {
        if (inputs.Length != Weights.Length) 
            throw new ArgumentException("Input count mismatch.");

        double sum = 0;
        for (int i = 0; i < inputs.Length; i++)
        {
            sum += inputs[i] * Weights[i];
        }
        sum += Bias;

        if (_config.UseReluActivation) 
            return Math.Max(0, sum);
            
        return sum;
    }

    public void Train(double[] inputs, double targetAnswer, double learningRate)
    {
        double guess = Predict(inputs);
        double error = targetAnswer - guess;

        for (int i = 0; i < Weights.Length; i++)
        {
            Weights[i] += error * inputs[i] * learningRate;
        }
        
        Bias += error * learningRate;
    }

    public void Save(string filePath)
    {
        using BinaryWriter writer = new(File.Open(filePath, FileMode.Create));
        writer.Write(Bias);
        writer.Write(Weights.Length);
        
        for (int i = 0; i < Weights.Length; i++)
        {
            writer.Write(Weights[i]);
        }
    }

    public void Load(string filePath)
    {
        using BinaryReader reader = new(File.Open(filePath, FileMode.Open));
        Bias = reader.ReadDouble();
        int expectedWeightCount = reader.ReadInt32();

        if (expectedWeightCount != Weights.Length)
        {
            throw new InvalidOperationException("Model architecture mismatch.");
        }

        for (int i = 0; i < expectedWeightCount; i++)
        {
            Weights[i] = reader.ReadDouble();
        }
    }
}