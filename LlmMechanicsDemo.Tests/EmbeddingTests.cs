using Xunit;
using LlmMechanicsDemo.Library;

namespace LlmMechanicsDemo.Tests;

public class EmbeddingTests
{
    [Fact]
    public void EmbeddingLayer_TranslatesIntegerIds_ToMeaningfulDecimalArrays()
    {
        // ARRANGE: Create a 3-dimensional conceptual map
        int dimensions = 3;
        var embeddingLayer = new EmbeddingLayer(dimensions);

        // Map dimensions: [Humanity, Royalty, Femininity]
        int kingId = 100;
        double[] kingCoordinates = [0.9, 0.9, -0.9];
        embeddingLayer.SetWordVector(kingId, kingCoordinates);

        int queenId = 101;
        double[] queenCoordinates = [0.9, 0.9, 0.9];
        embeddingLayer.SetWordVector(queenId, queenCoordinates);

        int appleId = 42;
        double[] appleCoordinates = [-0.9, -0.1, 0.0];
        embeddingLayer.SetWordVector(appleId, appleCoordinates);

        // ACT: The Tokenizer outputs the ID for "Queen", and we fetch its coordinates
        double[] retrievedVector = embeddingLayer.GetVector(101);

        // ASSERT: We now have the double[] array ready to feed into our NeuronLayer!
        Assert.Equal(3, retrievedVector.Length);
        Assert.Equal(0.9, retrievedVector[0]); // Human
        Assert.Equal(0.9, retrievedVector[1]); // Royalty
        Assert.Equal(0.9, retrievedVector[2]); // Female
    }
}