namespace LlmMechanicsDemo.Library;

public class SimpleTokenizer
{
    // The "Menu": Maps string chunks to integer IDs, and vice versa.
    private readonly Dictionary<string, int> _chunkToId;
    private readonly Dictionary<int, string> _idToChunk;

    public SimpleTokenizer()
    {
        _chunkToId = new Dictionary<string, int>();
        _idToChunk = new Dictionary<int, string>();
    }

    // Concept: Building the vocabulary menu
    public void AddToken(string chunk, int id)
    {
        if (_chunkToId.ContainsKey(chunk) || _idToChunk.ContainsKey(id))
        {
            throw new ArgumentException("Token or ID already exists in the vocabulary.");
        }

        _chunkToId.Add(chunk, id);
        _idToChunk.Add(id, chunk);
    }

    // Concept: Translating Human Text -> Network Numbers
    public int[] Encode(string[] textChunks)
    {
        int[] tokenIds = new int[textChunks.Length];

        for (int i = 0; i < textChunks.Length; i++)
        {
            string chunk = textChunks[i];
            
            if (_chunkToId.TryGetValue(chunk, out int id))
            {
                tokenIds[i] = id;
            }
            else
            {
                // If a word is completely unknown, it gets a special "Unknown" fallback ID (often 0).
                tokenIds[i] = 0; 
            }
        }

        return tokenIds;
    }

    // Concept: Translating Network Numbers -> Human Text
    public string Decode(int[] tokenIds)
    {
        string fullText = "";

        for (int i = 0; i < tokenIds.Length; i++)
        {
            int id = tokenIds[i];
            
            if (_idToChunk.TryGetValue(id, out string? chunk))
            {
                fullText += chunk;
            }
            else
            {
                fullText += "[UNK]";
            }
        }

        return fullText;
    }
}