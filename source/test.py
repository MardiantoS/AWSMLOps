import os
import boto3
import wget
import json
import sys
import numpy as np
from cv2 import imread, resize, IMREAD_GRAYSCALE


stack_name = sys.argv[1]
commit_id = sys.argv[2]
endpoint_name = f"{stack_name}-{commit_id[:7]}"

runtime = boto3.client("runtime.sagemaker")

# Image number is 7
IMAGE_URL = "https://mlopworkshop.s3.us-east-1.amazonaws.com/17.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBcaCXVzLWVhc3QtMSJIMEYCIQD%2FTaqAhEDmLG0JQgmOUSGiEWIGUjKc%2FvbwjgZUKpP7OAIhAP%2BkDlPqUrINrwGjd6h0O5fLOF1Yow4E3e0ep%2FmJympQKu4CCO%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FwEQARoMNTgyNDEzOTU3MzI0IgxNayHicFtvC%2BN514cqwgJwiv5NaMA9gjquYgFGIqU%2BbUDvGBCf8kL62ZCvvCzViqjqbGFcgXxbv2l7XLls0X%2FFUg01NNr15JFsbVJD%2Bnw4X6xSZBuFSv8S9VmzYtrc%2Fr9DXF%2BcH9wxjwGF5Z8LlR%2B53ne9HoJe%2BYqG%2F6J2y9ovOos%2FCkxtXJIL7xGhimUXICRZIA8spZ2xYnsbnolp%2BWe5%2BiJ98vdcu6ofi96IV3rBXHhiOCVxkx15qMpnNco%2BR7E6omzOix37Bs7M7pt%2Fy8zsCHUJtI8cVl7GTE1KlMhAo4NU0ZKMviaf3zsAqhC74gQxnqmOovvpcLDz7vdq5i5jNYUDiCWb6rFmeKD9sJ9gZc0yrd2w0v5Itb0nSd9g%2B8Czh1QfwZHr5IDEHj5cBOQmGk8ccdRAjWKMqoMhOPlk8aHCKL0cXO6VOa5smAlGZlM6MPigmZQGOoYCA5YbXcHnPg3wpC%2Fbx%2FYaYA7V6T%2Fo7KevudcyA8%2FV4JNlsKtFZ%2FIM%2BQQi%2BCZ6mil5ePh2e3OTTnu3g%2FwrFq4W8Uh7aAykIoNm30p6d0z11oe93zNCtQU36ftsiUFsyFLpGLCqZwxfyY46nRtB3SBUGYnEW7Fv8SOoCTqU6yN289ZJq%2BeB85%2F1p4W%2BuwkCVUL%2Fc979Rbqe8aE890Rb37M6LZezB6FBuq24rPRmoKugbXc94hcdGaigVM%2F%2FQbX%2Bw%2Fa%2FL4cmC0z2PRxKddFEh4LBdw56nHGGT6minzNJM5D0hMkvwj%2FcvEm7EHe80zDIWSWY7Md3Hp%2BheT8zF7G8SkTwf4FnKSErrA%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220519T142035Z&X-Amz-SignedHeaders=host&X-Amz-Expires=43200&X-Amz-Credential=ASIAYPGUTXDGAM6VI275%2F20220519%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=d79524f0e18faf3acac198d7c182b9f8f307751000df69a1e53e2cc0d1475834"
test_file = "test.jpg"
print(f'IMAGE_URL: {IMAGE_URL}')
wget.download(
    IMAGE_URL,
    test_file,
)

image = imread(test_file, IMREAD_GRAYSCALE)
image = resize(image, (28, 28))
image = image.astype("float32")
image = image.reshape(1, 1, 28, 28)

payload = json.dumps(image.tolist())
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name, Body=payload, ContentType="application/json"
)

result = response["Body"].read()
result = json.loads(result.decode("utf-8"))
print(f"Probabilities: {result}")

np_result = np.asarray(result)
prediction = np_result.argmax(axis=1)[0]
print(f"This is your number: {prediction}")

if prediction != 7:
    print("Model prediction failed.")
    sys.exit(1)

if os.path.exists(test_file):
    os.remove(test_file)
else:
    print("The file does not exist")
