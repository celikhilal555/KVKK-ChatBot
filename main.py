import requests
import json
class OllamaQA:
    def __init__(self, base_url='http://localhost:11434', model='llama3'):
        self.base_url = base_url
        self.model = model

    def generate_response(self, prompt):
        url = f"{self.base_url}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt
        }
        print("detay0", url)
        print(data)
        response = requests.post(url, json=data)
        full_response = ""
        for chunk in response.iter_lines():
            if chunk:
                data =json.loads(chunk)
                full_response += data["response"]
                if data["done"]:
                    break
        return full_response
    
    def chat(self):
        print("Ollama Q&A Bot is running. Type 'exit' to quit.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("Goodbye!")
                break
            response = self.generate_response(user_input)
            print(f"Ollama: {response}")

if __name__ == "__main__":
    ollama_qa = OllamaQA()
    ollama_qa.chat()