import configparser
import numpy as np
from my_rag.components.embeddings.aws_embedding import AWSBedrockEmbedding


def load_configuration(config_dir: str) -> dict:
    """Load configuration from Config.ini files"""
    config = configparser.ConfigParser()
    config.read(config_dir)
    return config


def test_bedrock_embedding():
    """
    Test the BedrockEmbedding class with actual API calls.
    This test verifies:
    1. Initialization of the embedding model
    2. Single text embedding
    4. Embedding with instruction
    5. Embedding dimensionality
    """
    # Load configuration containing AWS credentials
    config_dir = "config/config.ini"  # Adjust path as needed
    config = load_configuration(config_dir)

    # Initialize BedrockEmbedding
    embedding_model = AWSBedrockEmbedding(
        aws_access_key_id=config["BedRock_LLM_API"]["aws_access_key_id"],
        aws_secret_access_key=config["BedRock_LLM_API"]["aws_secret_access_key"],
        aws_session_token=config["BedRock_LLM_API"]["aws_session_token"],
        model_id="amazon.titan-embed-text-v2:0",
    )

    # Test 1: Single text embedding
    print("\nTest 1: Single text embedding")
    single_text = ["What is machine learning?"]
    single_embedding = embedding_model.embed(single_text)
    print(f"Single text embedding shape: {single_embedding.shape}")
    print(
        f"Embedding vector sample: {single_embedding[0][:5]}..."
    )  # Print first 5 values

    # Verify embedding properties
    assert isinstance(single_embedding, np.ndarray)
    assert len(single_embedding.shape) == 2
    assert single_embedding.shape[0] == 1
    print("✓ Single text embedding test passed")

    print("\nTest 2: Batch text embedding")
    batch_texts = [
        "What is deep learning?",
        "How does natural language processing work?",
        "Explain neural networks.",
        "What is reinforcement learning?",
    ]
    batch_embeddings = embedding_model.embed(batch_texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    print(f"Number of embeddings: {len(batch_embeddings)}")

    # Verify batch embedding properties
    assert isinstance(batch_embeddings, np.ndarray)
    assert len(batch_embeddings.shape) == 2
    assert batch_embeddings.shape[0] == len(batch_texts)
    print("✓ Batch text embedding test passed")

    # Test 3: Embedding with instruction
    print("\nTest 3: Embedding with instruction")
    instruction = "Represent this text for retrieval: "
    text_with_instruction = ["What are the main types of machine learning?"]
    instruction_embedding = embedding_model.embed(
        text_with_instruction, instruction=instruction
    )
    print(f"Instruction embedding shape: {instruction_embedding.shape}")
    print(f"Embedding vector sample: {instruction_embedding[0][:5]}...")

    # Verify instruction embedding properties
    assert isinstance(instruction_embedding, np.ndarray)
    assert len(instruction_embedding.shape) == 2
    assert instruction_embedding.shape[0] == 1
    print("✓ Instruction embedding test passed")

    # Test 4: Semantic similarity test
    print("\nTest 4: Semantic similarity test")
    similar_texts = [
        "What is artificial intelligence?",
        "Define AI and its applications.",
        "How to make coffee?",  # Dissimilar text
    ]
    similar_embeddings = embedding_model.embed(similar_texts)
    # Calculate cosine similarities
    similarities = np.dot(similar_embeddings, similar_embeddings.T)
    print("\nCosine similarity matrix:")
    print(similarities)

    # The first two texts should be more similar to each other than to the third
    assert similarities[0][1] > similarities[0][2]
    assert similarities[1][0] > similarities[1][2]
    print("✓ Semantic similarity test passed")
    # Cleanup
    embedding_model.clean_up()
    print("\n✓ All tests passed successfully!")


if __name__ == "__main__":
    test_bedrock_embedding()
