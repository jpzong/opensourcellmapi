import requests

class OllamaAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def download_model(self, model):
        endpoint = "/download_model/"
        url = self.base_url + endpoint
        try:
            response = requests.post(url, json={"model": model})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def text_to_text(self, model, prompt):
        endpoint = "/text-to-text/"
        url = self.base_url + endpoint
        try:
            response = requests.get(url, params={"model": model, "prompt": prompt})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

    def image_to_text(self, filename, prompt):
        endpoint = "/image-to-text/"
        url = self.base_url + endpoint
        try:
            files = {'filename': open(filename, 'rb')}
            response = requests.get(url, files=files, params={"prompt": prompt})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

# Example usage:
if __name__ == "__main__":
    api = OllamaAPI("http://localhost:8000")  
    
    print(api.download_model("my_model"))

    print(api.text_to_text("llama2", "Example prompt"))

    print(api.image_to_text("example_image.jpg", "Example prompt"))
