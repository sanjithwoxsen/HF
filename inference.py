from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("sanjithrj/FeelWise")
model = AutoModel.from_pretrained("sanjithrj/FeelWise",trust_remote_code=True)

input_text = "I am feeling happy today!"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
token_ids = inputs["input_ids"][0]  # Get the token IDs tensor
tokens = tokenizer.convert_ids_to_tokens(token_ids)
print(tokens)
print("Input :",inputs["input_ids"])
with torch.no_grad():
    outputs = model(inputs["input_ids"])
    print("output :",outputs)

# Apply softmax to logits to get probabilities
probs = F.softmax(outputs, dim=-1)

# Print the probabilities for each class
print("Probability :",probs)

# Get the predicted emotion class (the one with the highest probability)
predicted_emotion = torch.argmax(probs, dim=-1)

# Print the predicted emotion class index
print("predicted_emotion :",predicted_emotion)