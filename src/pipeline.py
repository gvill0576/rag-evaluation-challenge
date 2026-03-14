import boto3
import json
import chromadb

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

DOCUMENTS = [
    {"id": "doc1", "content": "Overfitting occurs when a model memorizes training data instead of learning generalizable patterns."},
    {"id": "doc2", "content": "To prevent overfitting, use regularization techniques like L1 or L2, or apply dropout layers."},
    {"id": "doc3", "content": "Neural networks learn through backpropagation, adjusting weights using gradients to minimize error."},
    {"id": "doc4", "content": "The learning rate controls how large each weight update step is during gradient descent."},
    {"id": "doc5", "content": "Cross-validation helps evaluate model generalization by testing on held-out data splits."},
    {"id": "doc6", "content": "Python is a popular programming language used in web development and scripting."},
    {"id": "doc7", "content": "Regularization adds a penalty term to the loss function to prevent overfitting by discouraging large weights."},
    {"id": "doc8", "content": "Backpropagation computes gradients layer by layer using the chain rule to update network weights."},
    {"id": "doc9", "content": "Cross-validation splits data into k folds, training on k-1 and validating on the remaining fold."},
    {"id": "doc10", "content": "The learning rate scheduler reduces the learning rate over time to improve model convergence."},
]

def create_embedding(text: str):
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": text})
    )
    return json.loads(response["body"].read())["embedding"]

class RAGPipeline:
    def __init__(self):
        client = chromadb.Client()
        self.vectorstore = client.get_or_create_collection(name="rag_docs")
        for doc in DOCUMENTS:
            self.vectorstore.add(
                ids=[doc["id"]],
                embeddings=[create_embedding(doc["content"])],
                documents=[doc["content"]]
            )

    def query(self, question: str) -> dict:
        results = self.vectorstore.query(
            query_embeddings=[create_embedding(question)],
            n_results=3
        )
        documents = [{"content": doc} for doc in results["documents"][0]]
        context = "\n".join(results["documents"][0])

        response = bedrock.converse(
            modelId="us.amazon.nova-lite-v1:0",
            messages=[{"role": "user", "content": [{"text": f"Answer based on context:\n\n{context}\n\nQuestion: {question}\n\nAnswer:"}]}],
            inferenceConfig={"maxTokens": 200, "temperature": 0.1}
        )
        answer = response["output"]["message"]["content"][0]["text"]

        return {"query": question, "answer": answer, "context": context, "documents": documents}
