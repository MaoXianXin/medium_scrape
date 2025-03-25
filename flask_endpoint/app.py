from flask import Flask, request, jsonify
from summarizer import summarize_article
import os
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
# 创建线程池来处理并发请求
executor = ThreadPoolExecutor(max_workers=10)

@app.route('/api/summarize', methods=['POST'])
def summarize_article_api():
    """
    API endpoint for summarizing articles using AI language models.
    
    This endpoint accepts POST requests with article text and optional configuration parameters.
    It processes the article asynchronously using a thread pool and returns a structured summary
    in Chinese Markdown format.
    
    Request (JSON):
        - article_text (str, required): The full text of the article to be summarized.
        - api_key (str, optional): API key for the language model service. 
                                  Defaults to OPENAI_API_KEY environment variable.
        - temperature (float, optional): Controls randomness in the model's output (0.0-1.0).
                                        Lower values make output more deterministic. Defaults to 0.1.
        - base_url (str, optional): Base URL for the API endpoint. Defaults to "https://zzzzapi.com/v1".
        - model (str, optional): The language model to use. Defaults to "gpt-4o-mini".
    
    Response (JSON):
        - success: {"summary": str} - The structured article summary in Chinese Markdown format.
        - error: {"error": str} - Error message if the request fails.
    
    Status Codes:
        - 200: Success
        - 400: Missing required parameters
        - 500: Server error during processing
    
    Example Request:
        Python example using requests library:
        
        ```python
        import requests
        
        url = "http://localhost:5000/api/summarize"
        headers = {"Content-Type": "application/json"}
        payload = {
            "article_text": "长文章内容...",
            "temperature": 0.2,
            "model": "gpt-4o"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        result = response.json()
        
        if response.status_code == 200:
            print("Summary:", result["summary"])
        else:
            print("Error:", result.get("error", "Unknown error"))
        ```
    """
    # 获取请求数据
    data = request.json
    
    # 检查必要参数
    if not data or 'article_text' not in data:
        return jsonify({"error": "Missing article_text parameter"}), 400
    
    # 获取参数
    article_text = data['article_text']
    api_key = data.get('api_key', os.environ.get('OPENAI_API_KEY'))
    temperature = data.get('temperature', 0.1)
    base_url = data.get('base_url', "https://zzzzapi.com/v1")
    model = data.get('model', "gpt-4o-mini")
    
    # 调用总结函数
    try:
        # 使用线程池异步处理请求
        future = executor.submit(
            summarize_article,
            article_text=article_text,
            api_key=api_key,
            temperature=temperature,
            base_url=base_url,
            model=model
        )
        
        summary = future.result()  # 等待结果
        return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 在生产环境中，你应该使用正确的WSGI服务器
    # 使用多线程模式运行Flask
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)