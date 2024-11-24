import requests
import json

def request_model(text, system_prompt="You are an AI assistant", max_new_tokens=1024, top_k=50, temperature=0.4, top_p=0.9):
    url = "http://192.168.90.253:11111/generate"
    data = {
        "text": text,
        "system_prompt": system_prompt,
        "max_new_tokens": max_new_tokens,
        "top_k": top_k,
        "temperature": temperature,
        "top_p": top_p
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        if response.status_code == 200:
            return response.json() 
        else:
            return f"Error: {response.status_code}, {response.text}"
    
    except requests.exceptions.RequestException as e:
        return f"Request failed: {str(e)}"

result = request_model("Hallo")[0]
print(result)
