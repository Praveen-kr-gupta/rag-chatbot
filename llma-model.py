import json
import boto3

prompt = """
who was the first pm of india ?
"""

bedrock = boto3.client(service_name = 'bedrock-runtime', region_name = 'us-east-1')

payload = {
    "prompt": prompt,
    "max_gen_len" : 512,
    "temperature":0.5,
    "top_p":0.9
}

body = json.dumps(payload)
model_id = "meta.llama3-70b-instruct-v1:0"

response = bedrock.invoke_model(body = body,
                                modelId = model_id,
                                contentType = "application/json",
                                accept= "application/json")

response_body = json.loads(response['body'].read())
print(response_body, "response")