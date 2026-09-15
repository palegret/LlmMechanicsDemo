using Xunit;
using LlmMechanicsDemo.Library;

namespace LlmMechanicsDemo.Tests;

public class EndToEndTests
{
    [Fact]
    public void FullPipeline_TranslatesTextToMath_AndGeneratesNextWord()
    {
        // ==========================================
        // PHASE 1: SYSTEM SETUP (Building the Car)
        // ==========================================
        
        // Map Dimensions: [Humanity, Royalty, Femininity]
        int dimensions = 3; 

        var tokenizer = new SimpleTokenizer();
        var embeddings = new EmbeddingLayer(dimensions);
        var outputProjection = new OutputProjectionLayer(dimensions);

        // 1a. Build the Vocabulary and the Map
        int royalId = 10;   double[] royalCoords = [0.0, 1.0, 0.0];
        int womanId = 11;   double[] womanCoords = [1.0, 0.0, 1.0];
        int queenId = 12;   double[] queenCoords = [1.0, 1.0, 1.0];
        int appleId = 13;   double[] appleCoords = [-1.0,-1.0, 0.0];

        tokenizer.AddToken("royal", royalId);
        tokenizer.AddToken(" woman", womanId);
        tokenizer.AddToken("queen", queenId);
        tokenizer.AddToken("apple", appleId);

        embeddings.SetWordVector(royalId, royalCoords);
        embeddings.SetWordVector(womanId, womanCoords);
        embeddings.SetWordVector(queenId, queenCoords);
        embeddings.SetWordVector(appleId, appleCoords);

        // In real models, the filing cabinet uses the same coordinates as the map.
        outputProjection.AddVocabularyWord(royalId, royalCoords);
        outputProjection.AddVocabularyWord(womanId, womanCoords);
        outputProjection.AddVocabularyWord(queenId, queenCoords);
        outputProjection.AddVocabularyWord(appleId, appleCoords);

        // 1b. Build the Deep Neural Network (The Engine)
        var config = new NetworkHyperparameters { InputCount = 3, NeuronCount = 3, UseReluActivation = false };
        var deepNetwork = new DeepNeuralNetwork([config]);

        // We manually tune this network to simply pass the coordinates through cleanly.
        var layer1 = deepNetwork.Layers[0];
        layer1.Neurons[0].Weights[0] = 1.0; layer1.Neurons[0].Weights[1] = 0.0; layer1.Neurons[0].Weights[2] = 0.0; // Preserves Humanity
        layer1.Neurons[1].Weights[0] = 0.0; layer1.Neurons[1].Weights[1] = 1.0; layer1.Neurons[1].Weights[2] = 0.0; // Preserves Royalty
        layer1.Neurons[2].Weights[0] = 0.0; layer1.Neurons[2].Weights[1] = 0.0; layer1.Neurons[2].Weights[2] = 1.0; // Preserves Femininity


        // ==========================================
        // PHASE 2: INFERENCE (Driving the Car)
        // ==========================================

        // Step 1: The user types a prompt.
        string[] prompt = ["royal", " woman"];

        // Step 2: Tokenizer translates English words to Integer IDs.
        int[] promptIds = tokenizer.Encode(prompt); // Returns [10, 11]

        // Step 3: Embeddings translate Integer IDs to Math Arrays (Vectors).
        double[] combinedContextVector = new double[dimensions];
        
        for (int i = 0; i < promptIds.Length; i++)
        {
            double[] wordVector = embeddings.GetVector(promptIds[i]);
            
            // We combine the words by adding their coordinates together.
            for (int d = 0; d < dimensions; d++)
            {
                combinedContextVector[d] += wordVector[d];
            }
        }
        
        // combinedContextVector is now [1.0, 1.0, 1.0]

        // Step 4: The Deep Neural Network crunches the math.
        // It applies its internal weights and biases to "think" about the context.
        double[] networkThought = deepNetwork.Process(combinedContextVector);

        // Step 5: Output Projection finds the closest matching word in the dictionary.
        int predictedTokenId = outputProjection.PredictMostLikelyNextToken(networkThought);

        // Step 6: Tokenizer translates the winning ID back to English.
        int[] finalIdArray = [predictedTokenId];
        string generatedWord = tokenizer.Decode(finalIdArray);

        // ==========================================
        // PHASE 3: VERIFICATION
        // ==========================================
        
        // The math successfully calculated that "royal" + "woman" = "queen".
        Assert.Equal("queen", generatedWord);
    }
}
