using Xunit;
using LlmMechanicsDemo.Library;

namespace LlmMechanicsDemo.Tests;

public class OutputProjectionTests
{
    [Fact]
    public void OutputProjection_FindsTheClosestMatchingWord_FromNetworkOutput()
    {
        // ARRANGE: Set up the detective's filing cabinet (3 dimensions)
        
        int dimensions = 3;
        var projectionLayer = new OutputProjectionLayer(dimensions);

        // Dimensions: [Humanity, Royalty, Femininity]

        int kingId = 100;
        double[] kingCoords = [0.9, 0.9, -0.9];

        int queenId = 101;
        double[] queenCoords = [0.9, 0.9, 0.9];

        int appleId = 42;
        double[] appleCoords = [-0.9, -0.1, 0.0];

        projectionLayer.AddVocabularyWord(kingId, kingCoords);
        projectionLayer.AddVocabularyWord(queenId, queenCoords);
        projectionLayer.AddVocabularyWord(appleId, appleCoords);

        // ACT: The Deep Neural Network finishes its math and outputs a 
        // "Sketch". It's looking for something highly human, highly royal, 
        // and somewhat female.

        double[] networkSketch = [0.8, 0.8, 0.5];

        int winningTokenId = projectionLayer.PredictMostLikelyNextToken(networkSketch);

        // ASSERT: Let's do the manual math to prove why Queen wins.
        // King Score:  (0.8 * 0.9) + (0.8 * 0.9) + (0.5 * -0.9) = 0.72 + 0.72 - 0.45 = 0.99
        // Queen Score: (0.8 * 0.9) + (0.8 * 0.9) + (0.5 *  0.9) = 0.72 + 0.72 + 0.45 = 1.89
        // Apple Score: (0.8 * -0.9)+ (0.8 * -0.1)+ (0.5 *  0.0) = -0.72 - 0.08 + 0.0 = -0.80

        // The ID for "Queen" wins the highest score!
        Assert.Equal(queenId, winningTokenId); 
    }
}
