import ollama

def label_data(data):
    """Send data to TinyLlama for labeling."""
    prompt = f"Label this dataset based on sensor readings:\n{data.to_json()}"
    response = ollama.chat(model="tinyllama", messages=[{"role": "user", "content": prompt}])
    return response['message']['content']
