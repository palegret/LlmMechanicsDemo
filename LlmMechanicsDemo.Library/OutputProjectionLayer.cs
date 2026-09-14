namespace LlmMechanicsDemo.Library;

public class OutputProjectionLayer(int vectorDimensions)
{
    // The "Filing Cabinet" of mugshots: Maps Token IDs to their coordinate 
    // vectors. In real models, this is often the exact same matrix used in 
    // the EmbeddingLayer!
    private readonly Dictionary<int, double[]> _vocabularyEmbeddings = [];
    private readonly int _vectorDimensions = vectorDimensions;

    public void AddVocabularyWord(int tokenId, double[] coordinates)
    {
        if (coordinates.Length != _vectorDimensions)
        {
            throw new ArgumentException("Coordinate array must match the defined dimensions.");
        }

        _vocabularyEmbeddings[tokenId] = coordinates;
    }

    // Concept: The Detective comparing the sketch against the filing cabinet.
    public int PredictMostLikelyNextToken(double[] networkFinalOutput)
    {
        if (networkFinalOutput.Length != _vectorDimensions)
        {
            throw new ArgumentException("Network output must match the embedding dimensions.");
        }

        int bestTokenId = -1;
        double highestScore = double.MinValue;

        // Loop through every single word in the vocabulary
        foreach (var kvp in _vocabularyEmbeddings)
        {
            int currentTokenId = kvp.Key;
            double[] wordCoordinates = kvp.Value;

            // Calculate the similarity (Dot Product)
            double matchScore = 0;

            for (int i = 0; i < _vectorDimensions; i++)
            {
                matchScore += networkFinalOutput[i] * wordCoordinates[i];
            }

            // Keep track of the highest scorer
            
            if (matchScore > highestScore)
            {
                highestScore = matchScore;
                bestTokenId = currentTokenId;
            }
        }

        return bestTokenId;
    }
}
