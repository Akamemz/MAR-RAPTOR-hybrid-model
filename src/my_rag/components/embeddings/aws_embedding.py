import boto3
from typing import List, Optional
import numpy as np
import json
from botocore.exceptions import ClientError
from .base import BaseEmbedding


class AWSBedrockEmbedding(BaseEmbedding):
    """
    Embedding model using AWS Bedrock.
    """

    def __init__(
        self,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_session_token: Optional[str] = None,
        region_name: str = "us-east-1",
        model_id: str = "amazon.titan-embed-text-v2:0",
        **kwargs,
    ):
        """
        Initializes the AWS Bedrock embedding model.

        Args:
            aws_access_key_id (str): AWS access key ID
            aws_secret_access_key (str): AWS secret access key
            aws_session_token (Optional[str]): AWS session token
            region_name (str): AWS region name
            model_id (str): Bedrock model ID for embeddings
            **kwargs: Additional keyword arguments
        """
        self.model_id = model_id

        try:
            self.session = boto3.Session(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
            )
            self.client = self.session.client(
                "bedrock-runtime", region_name=region_name
            )
        except Exception as e:
            raise ValueError(f"Failed to initialize Bedrock client: {str(e)}")

    def embed(
        self,
        texts: List[str],
        instruction: Optional[str] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Embeds a list of texts using AWS Bedrock.

        Args:
            texts (List[str]): The texts to embed
            instruction (Optional[str]): Instruction text (not used for Bedrock)
            **kwargs: Additional keyword arguments

        Returns:
            np.ndarray: Array of embeddings
        """
        all_embeddings = []

        for text in texts:
            if instruction:
                text = f"{instruction} {text}"
            try:
                # Prepare request body
                body_content = {"inputText": text}

                # Call Bedrock API
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType="application/json",
                    accept="application/json",
                    body=json.dumps(body_content),
                )

                # Parse response
                response_body = json.loads(response["body"].read())
                embedding = response_body["embedding"]
                all_embeddings.append(embedding)

            except ClientError as e:
                raise Exception(f"Bedrock API error: {str(e)}")
            except Exception as e:
                raise Exception(f"Error generating embedding: {str(e)}")

        all_embeddings = np.array(all_embeddings)
        # # Combine all batches
        # embeddings = np.vstack(all_embeddings)

        # # Normalize embeddings
        # embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        return all_embeddings

    def clean_up(self):
        """
        Cleans up resources.
        """
        if hasattr(self, "client"):
            self.client.close()
