# Banana Python SDK

### Getting Started

Install via pip
`pip3 install aiobananas`

Get your API Key
- [Sign in / log in here](https://app.banana.dev)

Run:
```python
import aiobananas

api_key = "demo" # "YOUR_API_KEY"
model_key = "carrot" # "YOUR_MODEL_KEY"
model_inputs = {
    # a json specific to your model. For example:
    "imageURL":  "https://demo-images-banana.s3.us-west-1.amazonaws.com/image2.jpg"
}

async with aiobananas.Session(api_key) as banana:
    out = await banana.run(model_key, model_inputs)



out = banana.run(api_key, model_key, model_inputs)
print(out)
```

Return type:
```python
{
    "id": "12345678-1234-1234-1234-123456789012", 
    "message": "success", 
    "created": 1649712752, 
    "apiVersion": "26 Nov 2021", 
    "modelOutputs": [
        {
            # a json specific to your model. In this example, the caption of the image
            "caption": "a baseball player throwing a ball"
        }
    ]
}
```

Parse the server output:
```python
model_out = out["modelOutputs"][0]
```