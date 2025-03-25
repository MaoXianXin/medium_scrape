import requests
import json
import os
import time
import concurrent.futures
from pathlib import Path

def extract_main_points(article_summary, api_key=None, base_url="https://zzzzapi.com/v1", model="gemini-2.0-flash", temperature=0.2, return_tokens=True):
    # API 端点
    url = "http://localhost:5000/api/extract_main_points"
    
    # 请求参数
    payload = {
        "article_summary": article_summary,
        "api_key": api_key,
        "return_tokens": return_tokens,
        "temperature": temperature,
        "base_url": base_url,
        "model": model,
    }
    
    # 发送POST请求
    response = requests.post(url, json=payload)
    
    return response

def process_summary(summary_data, api_key=None, base_url="https://zzzzapi.com/v1", model="gemini-2.0-flash", temperature=0.2, return_tokens=True):
    """处理单个摘要并返回结果"""
    try:
        file_name = summary_data.get('file')
        article_summary = summary_data.get('summary')
        
        if not article_summary:
            return {
                'file': file_name,
                'status': 'error',
                'error': 'Missing summary in input data'
            }
        
        start_time = time.time()
        response = extract_main_points(article_summary, api_key, base_url, model, temperature, return_tokens)
        end_time = time.time()
        
        if response.status_code == 200:
            result = response.json()
            return {
                'file': file_name,
                'status': 'success',
                'time': end_time - start_time,
                'start_time': start_time,
                'end_time': end_time,
                'main_points': result.get('main_points'),
                'token_info': result.get('token_info')
            }
        else:
            return {
                'file': file_name,
                'status': 'error',
                'time': end_time - start_time,
                'start_time': start_time,
                'end_time': end_time,
                'error': response.text
            }
    except Exception as e:
        return {
            'file': file_name if file_name else 'unknown',
            'status': 'exception',
            'error': str(e)
        }

def concurrent_test_from_results(results_file_path, max_workers=5, api_key=None, base_url="https://zzzzapi.com/v1", model="gemini-2.0-flash", temperature=0.2, return_tokens=True):
    """从之前的摘要结果文件中并发测试核心观点提取"""
    try:
        with open(results_file_path, 'r', encoding='utf-8') as f:
            summary_results = json.load(f)
    except Exception as e:
        print(f"错误: 无法加载结果文件 {results_file_path}: {str(e)}")
        return
    
    if not summary_results:
        print(f"警告: {results_file_path} 中没有找到摘要结果")
        return
    
    # 只处理成功的摘要
    valid_summaries = [r for r in summary_results if r.get('status') == 'success' and r.get('summary')]
    
    print(f"找到 {len(valid_summaries)} 个有效摘要，开始并发测试核心观点提取...")
    
    results = []
    total_start_time = time.time()
    
    # 使用线程池并发处理摘要
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_summary = {executor.submit(process_summary, summary_data, api_key, base_url, model, temperature, return_tokens): summary_data for summary_data in valid_summaries}
        
        for future in concurrent.futures.as_completed(future_to_summary):
            summary_data = future_to_summary[future]
            try:
                result = future.result()
                results.append(result)
                print(f"处理完成: {result.get('file')} - 状态: {result['status']}")
            except Exception as e:
                print(f"处理 {summary_data.get('file', 'unknown')} 时发生异常: {str(e)}")
    
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
        total_processing_time = 0
        concurrency_efficiency = 0
    
    print("\n测试结果摘要:")
    print(f"总摘要数: {len(valid_summaries)}")
    print(f"成功处理: {success_count}")
    print(f"处理失败: {error_count}")
    print(f"平均处理时间: {avg_time:.2f} 秒")
    print(f"总测试时间: {total_end_time - total_start_time:.2f} 秒")
    print(f"理论串行处理总时间: {total_processing_time:.2f} 秒")
    print(f"并发效率: {concurrency_efficiency:.2f}x")
    print(f"有效并发数: {max(1, round(concurrency_efficiency))}")
    
    # 创建带有时间戳的文件名
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    result_filename = f"extract_points_test_{model}_{timestamp}_w{max_workers}.json"
    
    # 保存详细结果到JSON文件
    with open(result_filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"详细结果已保存到 {result_filename}")
    
    return results

# 运行示例
if __name__ == "__main__":
    # 从之前的摘要结果文件中并发测试核心观点提取
    results_file_path = "summary_test_gemini-2.0-flash_20250325_104654_w10.json"
    api_key = "sk-UxCneocSvk83jPkSmDRyYZA2zLWiAX1Ds71JVK72IqH1DiR6"
    base_url = "https://zzzzapi.com/v1"
    model = "gemini-2.0-flash"
    temperature = 0.2
    return_tokens = True
    max_workers = 10
    
    concurrent_test_from_results(
        results_file_path=results_file_path,
        max_workers=max_workers,
        api_key=api_key,
        base_url=base_url,
        model=model,
        temperature=temperature,
        return_tokens=return_tokens
    )