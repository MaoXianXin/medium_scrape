import requests
import os

# 从txt文件读取文章内容
with open('/home/mao/workspace/medium_scrape/articles/4-bit-quantization-with-gptq-36b0f4f02c34.txt', 'r', encoding='utf-8') as file:
    article_content = file.read()

# 第一步：调用summarize API获取文章摘要
summarize_response = requests.post(
    "http://localhost:5000/api/summarize",
    json={
        "article_text": article_content,
        "api_key": "sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
        "return_tokens": True,
        "temperature": 0.2,
        "base_url": "https://www.gptapi.us/v1",
        "model": "deepseek-r1",
    }
)

summarize_result = summarize_response.json()
print("摘要结果:")
print(summarize_result)

# 提取summary并保存到markdown文件
if 'summary' in summarize_result:
    # 创建保存目录（如果不存在）
    output_dir = '/home/mao/workspace/medium_scrape/summaries'
    os.makedirs(output_dir, exist_ok=True)
    
    # 从原文件名提取文章ID作为输出文件名
    article_id = '4-bit-quantization-with-gptq-36b0f4f02c34'
    summary_file = os.path.join(output_dir, f"{article_id}_summary.md")
    
    # 保存summary到markdown文件
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(summarize_result['summary'])
    
    print(f"Summary saved to {summary_file}")
    
    # 第二步：调用extract_main_points API提取核心观点
    extract_response = requests.post(
        "http://localhost:5000/api/extract_main_points",
        json={
            "article_summary": summarize_result['summary'],
            "api_key": "sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
            "return_tokens": True,
            "temperature": 0.2,
            "base_url": "https://www.gptapi.us/v1",
            "model": "deepseek-r1",
        }
    )
    
    extract_result = extract_response.json()
    print("\n核心观点提取结果:")
    print(extract_result)
    
    # 保存核心观点到markdown文件
    if 'main_points' in extract_result:
        main_points_file = os.path.join(output_dir, f"{article_id}_main_points.md")
        
        # 将列表转换为字符串格式
        main_points_text = "\n\n".join([f"- {point}" for point in extract_result['main_points']])
        
        with open(main_points_file, 'w', encoding='utf-8') as f:
            f.write(main_points_text)
        
        print(f"Main points saved to {main_points_file}")
    else:
        print("No main points found in the API response")
else:
    print("No summary found in the API response")