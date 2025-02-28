from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import os

load_dotenv()
api_key1 = os.environ.get('HUGGING_FACE_API_KEY') # Replace with your actual API token

# Replace 'model_name' with the desired model ID, e.g., "meta-llama/Llama-2-13B-chat-hf"
model_name = "mistralai/Mistral-7B-Instruct-v0.3"

client = InferenceClient(token=api_key1,model=model_name)

# Example input for the model
##input_text = "Describe a scene from Mahabharata."
##response = client(inputs=input_text)
##response = client(inputs="Describe a scene from Mahabharata.")
#print(response)
#input_text = "Describe a scene from Mahabharata."
#response = client.post(json={"inputs": input_text})
#print(response.json())  # Print the JSON response
# Prepare the input message as a list of messages
messages = [{"role": "user", "content": "Describe a scene from Mahabharata."}]

# Call the chat_completion method with your messages
response = client.chat_completion(model=model_name, messages=messages)

# Print the response content
print(response['choices'][0]['message']['content'])


