using Xunit;
using LlmMechanicsDemo.Library;

namespace LlmMechanicsDemo.Tests;

public class TokenizerTests
{
    [Fact]
    public void Tokenizer_TranslatesSubWords_ToIntegersAndBack()
    {
        // ARRANGE: Build the vocabulary menu.
        var tokenizer = new SimpleTokenizer();
        
        tokenizer.AddToken("<|start|>", 99); // Hidden control token
        tokenizer.AddToken("un", 100);
        tokenizer.AddToken("believ", 101);
        tokenizer.AddToken("ably", 102);
        tokenizer.AddToken(" fast", 103);     // Notice the leading space is part of the token
        tokenizer.AddToken("<|end|>", 104);   // Hidden control token

        // The user types: "unbelievably fast"
        // The chopping algorithm splits it into distinct chunks:
        string[] choppedInput = ["<|start|>", "un", "believ", "ably", " fast", "<|end|>"];

        // ACT: Encode to numbers for the neural network.
        int[] encodedNetworkInput = tokenizer.Encode(choppedInput);

        // ASSERT: Verify the exact numerical array the network will receive.
        Assert.Equal(6, encodedNetworkInput.Length);
        Assert.Equal(99, encodedNetworkInput[0]);
        Assert.Equal(100, encodedNetworkInput[1]);
        Assert.Equal(101, encodedNetworkInput[2]);
        Assert.Equal(102, encodedNetworkInput[3]);
        Assert.Equal(103, encodedNetworkInput[4]);
        Assert.Equal(104, encodedNetworkInput[5]);

        // ACT: The network finishes its math and outputs these IDs. Decode them back to English.
        int[] networkOutputIds = [100, 101, 102, 103];
        string humanReadableOutput = tokenizer.Decode(networkOutputIds);

        // ASSERT: Verify the numbers perfectly reconstruct the string.
        Assert.Equal("unbelievably fast", humanReadableOutput);
    }
}
