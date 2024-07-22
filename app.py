from inference_sdk import InferenceHTTPClient

# Create an inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="0Ns0ib19K5V6gNixQqST"
)

# Run inference on a local image
result = CLIENT.infer(
    image_path="C:\Users\HP-INDIA\Downloads\smartobjectidentifier.v2i.yolokeras.zip\train",  
    model_id="smartobjectidentifier/2"
)

# Print the result
print(result)
