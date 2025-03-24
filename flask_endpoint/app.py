from flask import Flask, request, jsonify
from article_summarizer import summarize_article
import os
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
# 创建线程池来处理并发请求
executor = ThreadPoolExecutor(max_workers=10)

@app.route('/api/summarize', methods=['POST'])
def api_summarize_article():
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
    return_tokens = data.get('return_tokens', False)
    
    # 调用总结函数
    try:
        # 使用线程池异步处理请求
        future = executor.submit(
            summarize_article,
            article_text=article_text,
            api_key=api_key,
            return_tokens=return_tokens,
            temperature=temperature,
            base_url=base_url,
            model=model
        )
        
        result = future.result()  # 等待结果
        
        if return_tokens:
            summary, token_info = result
            return jsonify({
                "summary": summary,
                "token_info": token_info
            })
        else:
            summary = result
            return jsonify({"summary": summary})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 在生产环境中，你应该使用正确的WSGI服务器
    # 使用多线程模式运行Flask
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)