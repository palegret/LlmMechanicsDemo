namespace LlmMechanicsDemo.Library;

public class EmbeddingLayer(int vectorDimensions)
{
    // The matrix maps an integer ID to an array of decimals (the coordinates)
    private readonly Dictionary<int, double[]> _embeddingMatrix = [];
    private readonly int _vectorDimensions = vectorDimensions;

    // During training, the network adjusts these coordinates. 
    // For this demo, we manually place the words on the map.
    public void SetWordVector(int tokenId, double[] coordinates)
    {
        if (coordinates.Length != _vectorDimensions)
        {
            throw new ArgumentException("Coordinate array must match the defined dimensions.");
        }
        
        _embeddingMatrix[tokenId] = coordinates;
    }

    // Translates the Tokenizer's integer ID into the math-ready decimal array
    public double[] GetVector(int tokenId)
    {
        if (_embeddingMatrix.TryGetValue(tokenId, out double[]? coordinates))
        {
            return coordinates;
        }
        
        // If the token has no map coordinates, return an array of zeros.
        return new double[_vectorDimensions];
    }
}