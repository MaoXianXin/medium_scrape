import requests

# Read article text from a file
try:
    with open("/home/mao/workspace/medium_scrape/articles/4-bit-quantization-with-gptq-36b0f4f02c34.txt", "r", encoding="utf-8") as file:
        article_text = file.read()
except FileNotFoundError:
    print("Error: article.txt file not found.")
    exit(1)
except Exception as e:
    print(f"Error reading file: {e}")
    exit(1)

url = "http://localhost:5000/api/summarize"
headers = {"Content-Type": "application/json"}
payload = {
    "article_text": article_text,
    "temperature": 0.2,
    "model": "deepseek-r1",
    "api_key": "sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
    "base_url": "https://www.gptapi.us/v1"
}

response = requests.post(url, headers=headers, json=payload)
result = response.json()

if response.status_code == 200:
    print("Summary:", result["summary"])
else:
    print("Error:", result.get("error", "Unknown error"))