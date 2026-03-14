import boto3

bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')

def invoke_bedrock(prompt: str, max_tokens: int = 200, temperature: float = 0.1) -> str:
    response = bedrock.converse(
        modelId="us.amazon.nova-lite-v1:0",
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature}
    )
    return response["output"]["message"]["content"][0]["text"]

def parse_score(response: str, default: int = 5) -> int:
    try:
        if "SCORE:" in response:
            score_part = response.split("SCORE:")[1].split("\n")[0]
            digits = ''.join(filter(str.isdigit, score_part.split("/")[0]))
            return min(10, max(0, int(digits)))
    except:
        pass
    return default

def measure_faithfulness(context: str, answer: str) -> dict:
    prompt = f"""Rate how well this answer is supported by the context.

Context:
{context}

Answer:
{answer}

Score 0-10 where 10 means every claim is directly supported.

Respond in this format:
SCORE: [0-10]
SUPPORTED: [Claims found in context]
UNSUPPORTED: [Claims not in context, or "None"]
REASONING: [One sentence]"""

    response = invoke_bedrock(prompt)
    return {"score": parse_score(response), "max_score": 10, "raw": response}

def measure_relevance(query: str, answer: str) -> dict:
    prompt = f"""Rate how well this answer addresses the question.

Question: {query}
Answer: {answer}

Score 0-10 where 10 means directly and completely answers the question.

Respond in this format:
SCORE: [0-10]
ADDRESSED: [Parts of question answered]
MISSED: [Parts not answered, or "None"]
REASONING: [One sentence]"""

    response = invoke_bedrock(prompt)
    return {"score": parse_score(response), "max_score": 10, "raw": response}

def measure_precision(query: str, documents: list) -> dict:
    relevant = 0
    for doc in documents:
        prompt = f"""Is this document relevant to the question? Answer YES or NO only.

Question: {query}
Document: {doc['content']}"""
        response = invoke_bedrock(prompt, max_tokens=10)
        if "YES" in response.upper() and "NO" not in response.upper():
            relevant += 1

    precision = relevant / len(documents) if documents else 0
    return {
        "score": round(precision * 10),
        "max_score": 10,
        "relevant": relevant,
        "total": len(documents),
        "precision": precision
    }
