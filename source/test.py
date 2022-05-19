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
IMAGE_URL = "https://mlopworkshop.s3.us-east-1.amazonaws.com/17.png?response-content-disposition=inline&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEBoaCXVzLWVhc3QtMSJGMEQCIAZh7MbY8HQ4bkTY7OEkSNwUUzcXccqoPrnaC0Q5iucUAiBgb0MSwFy%2BKT1R1aFStjp%2Bl3cyr5twGwdLIb3eU0E%2BKiruAgjy%2F%2F%2F%2F%2F%2F%2F%2F%2F%2F8BEAEaDDU4MjQxMzk1NzMyNCIMq46kElYy4Vb5oXUgKsICfgQP%2BwXJkYOEOOXwYm8HG3MbJcNEiMeBRoR9aokghLOw%2Baug2poKP%2FhCFlqIAFAmjMLlbTqwwfvjVSyCLhxHLrOKUx6zD96c2ldgQUdx78OxWPAJ24vX7SPsWULfioHaJPmnzyLf3kQ5oe%2FgtxPLuxPje1sbS%2FBV%2FR6c7AYZr822VKOw%2BfS4bubUrrW1DZkub8Fl6%2FyXkuW77pVMsLCFLbzc6GQpSJmy5yoW%2BEyQpvCQ%2BhRTYSTiDTpUXGg1KIlLG3f3vr56tbWwqF5Jdm0w9wuQixcvljvFYS3CnzH4BDywPXxrJAFWZMjjGZc2%2FIRG0UU9uvXboBQXNva7pp9yYvDs2Us51PX7IMgPF9X3xxm0gPl3tWUWKhzvmUQ6yRGD2Rmf0IjdWgB86Tpj%2BCkhG7NRpDAmEwAl8v5tM9Bx51VDtTCN%2BZmUBjqIAgB6FlnA37u9YL%2BUzbicGKTzWAhTtKPC0ZVs45%2BqPD0QqykxKenDmQXjYhMYmFcyZOEIEoiP1emVl%2Ft0%2BEeeHcnU3OMcSfyhVC%2Bmsr4GDMRQWMY%2FtKtC90TVBkZZbtMDD1wzhwMMyuOrHXV%2B1Qc0X9wCR%2FYHvm3MgB4PjlBrPFLOO4uN%2BME5q%2B13iRpTqx2H1WhBfaYdrp1zd7zM6ZQZSlB7GQB6oTMWW2vR2LsKIUjiSZcXQ816yRdM7U%2FL7GrgFKZBj17KcqFkkbE4Ky1DCuiLf8kUVrtV%2BYI4tcw9%2BFueCalmKpX%2FbLfGGMbMe%2Blh1U9DCyxoqug79ICpL4k99V%2F%2BrnaHtcY1Aw%3D%3D&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Date=20220519T172205Z&X-Amz-SignedHeaders=host&X-Amz-Expires=36000&X-Amz-Credential=ASIAYPGUTXDGG6DGX2VJ%2F20220519%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Signature=715a2db2bc298f207de097c1969fbbf3c15e92e82f0165fe1da5c9ba179d25d0"
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
