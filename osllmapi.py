import requests

class OllamaAPI:
    def __init__(self, base_url):
        self.base_url = base_url

    def download_model(self, model):
        endpoint = "/download_model/"
        url = self.base_url + endpoint
        try:
            response = requests.post(url, params={"model": model})
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
            print(filename)
            files = {'file': open(filename, 'rb')}
            response = requests.post(url, files=files, params={"prompt": prompt})
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {"error": str(e)}

"""
if __name__ == "__main__":
    api = OllamaAPI("http://213.173.108.19:14301")  
    
    #print(api.download_model("llama2"))

    print(api.text_to_text("mixtral", "Cuentame un chiste"))

    #print(api.image_to_text("male-fashion-model.jpg", "De qué color es la playera?"))
"""
