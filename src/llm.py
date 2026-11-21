import boto3
import json

client = boto3.client(
    "bedrock-runtime",
    region_name="us-east-1"
)


prompt=input("Enter your prompt: ")
response = client.invoke_model(
    modelId="amazon.nova-micro-v1:0",
    body=json.dumps({
        "messages": [
            {
                "role": "user",
                "content": [{"text": prompt}]
            }
        ],
        "inferenceConfig": {"maxTokens": 64}
    })
)

result = json.loads(response["body"].read())
print(result["output"]["message"]["content"][0]["text"])

print("--"*30)
print(f"Input tokens: {result['usage']['inputTokens']}")
print(f"Output tokens: {result['usage']['outputTokens']}")
print(f"Total tokens: {result['usage']['totalTokens']}")