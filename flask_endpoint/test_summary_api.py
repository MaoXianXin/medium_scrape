import requests
import json
import os
import time
import concurrent.futures
from pathlib import Path

def summarize_article(article_text, api_key=None, base_url="https://zzzzapi.com/v1", model="gemini-2.0-flash", temperature=0.2, return_tokens=True):
    # API 端点
    url = "http://localhost:5000/api/summarize"
    
    # 请求参数
    payload = {
        "article_text": article_text,
        "api_key": api_key,
        "return_tokens": return_tokens,
        "temperature": temperature,
        "base_url": base_url,
        "model": model,
    }
    
    # 发送POST请求
    response = requests.post(url, json=payload)
    
    return response

def process_file(file_path, api_key=None, base_url="https://zzzzapi.com/v1", model="gemini-2.0-flash", temperature=0.2, return_tokens=True):
    """处理单个文件并返回结果"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            article_text = f.read()
        
        start_time = time.time()
        response = summarize_article(article_text, api_key, base_url, model, temperature, return_tokens)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                'file': file_path.name,
                'status': 'success',
                'time': end_time - start_time,
                'start_time': start_time,
                'end_time': end_time,
                'summary': result.get('summary'),
                'token_info': result.get('token_info')
            }
        else:
            return {
                'file': file_path.name,
                'status': 'error',
                'time': end_time - start_time,
                'start_time': start_time,
                'end_time': end_time,
                'error': response.text
            }
    except Exception as e:
        return {
            'file': file_path.name,
            'status': 'exception',
            'error': str(e)
        }

def concurrent_test(directory_path, max_workers=5, api_key=None, base_url="https://zzzzapi.com/v1", model="gemini-2.0-flash", temperature=0.2, return_tokens=True):
    """并发测试指定目录下的所有txt文件"""
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        print(f"错误: {directory_path} 不是有效目录")
        return
    
    # 获取所有txt文件
    txt_files = list(directory.glob('*.txt'))
    if not txt_files:
        print(f"警告: {directory_path} 中没有找到txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件，开始并发测试...")
    
    results = []
    total_start_time = time.time()
    
    # 添加请求开始时间戳到结果中
    request_timestamps = {}
    
    # 使用线程池并发处理文件
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {executor.submit(process_file, file_path, api_key, base_url, model, temperature, return_tokens): file_path for file_path in txt_files}
        
        for future in concurrent.futures.as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f"处理完成: {file_path.name} - 状态: {result['status']}")
            except Exception as e:
                print(f"处理 {file_path.name} 时发生异常: {str(e)}")
    
    total_end_time = time.time()
    
    # 统计结果
    success_count = sum(1 for r in results if r['status'] == 'success')
    error_count = len(results) - success_count
    avg_time = sum(r.get('time', 0) for r in results if 'time' in r) / len(results) if results else 0
    
    # 计算并发度指标
    if success_count > 0:
        # 计算理论串行时间 (所有请求处理时间之和)
        total_processing_time = sum(r.get('time', 0) for r in results if 'time' in r)
        # 实际总时间
        actual_total_time = total_end_time - total_start_time
        # 并发效率 (理论串行时间/实际时间)
        concurrency_efficiency = total_processing_time / actual_total_time if actual_total_time > 0 else 0
    else:
        concurrency_efficiency = 0
    
    print("\n测试结果摘要:")
    print(f"总文件数: {len(txt_files)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {error_count}")
    print(f"平均处理时间: {avg_time:.2f} 秒")
    print(f"总测试时间: {total_end_time - total_start_time:.2f} 秒")
    print(f"理论串行处理总时间: {total_processing_time:.2f} 秒")
    print(f"并发效率: {concurrency_efficiency:.2f}x")
    print(f"有效并发数: {max(1, round(concurrency_efficiency))}")
    
    # 创建带有时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f"summary_test_{model}_{timestamp}_w{max_workers}.json"
    
    # 保存详细结果到JSON文件
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存到 {result_filename}")
    
    return results

# 运行示例
if __name__ == "__main__":
    # 并发测试示例
    # 替换为你的txt文件目录路径
    directory_path = "/home/mao/workspace/medium_scrape/111/"
    api_key = "sk-UxCneocSvk83jPkSmDRyYZA2zLWiAX1Ds71JVK72IqH1DiR6"
    base_url = "https://zzzzapi.com/v1"
    model = "gemini-2.0-flash"
    temperature = 0.2
    return_tokens = True
    max_workers = 10
    
    concurrent_test(
        directory_path=directory_path,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        return_tokens=return_tokens
    )