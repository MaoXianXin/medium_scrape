from flask_endpoint.dialog_module.base import OneTimeDialogModule
from flask_endpoint.dialog_module.utils import read_template_from_file, read_article_from_file, create_custom_llm
import re
import os
import json
import hashlib

"""
python -m flask_endpoint.data_preparation.generate_summary_labels
"""

def summarize_article(article_path, template_path):
    """
    使用LLM对文章进行总结
    
    Args:
        article_path: 文章文件路径
        template_path: 总结提示词模板文件路径
        
    Returns:
        文章总结内容
    """
    # 读取文章内容
    article_content = read_article_from_file(article_path)
    if article_content == "无法读取文件内容":
        return "无法读取文章内容，请检查文件路径"
    
    # 读取提示词模板
    template = read_template_from_file(template_path)
    if template == "无法读取模板文件内容":
        return "无法读取提示词模板，请检查模板文件路径"
    
    # 创建LLM实例
    llm = create_custom_llm(model_name="gpt-4.1-nano-2025-04-14", base_url="https://zzzzapi.com/v1", api_key="sk-UxCneocSvk83jPkSmDRyYZA2zLWiAX1Ds71JVK72IqH1DiR6")
    
    # 创建对话模块
    dialog_module = OneTimeDialogModule(
        llm=llm,
        prompt_template=template,
        template_variables={"article_text": article_content}
    )
    
    # 处理并获取总结
    summary = dialog_module.process()
    
    return summary

def process_directory(directory_path, template_path, output_json_path):
    """
    处理指定目录下的所有txt文件并保存对话记录到JSON
    
    Args:
        directory_path: 包含txt文件的目录路径
        template_path: 总结提示词模板文件路径
        output_json_path: 输出JSON文件路径
    """
    # 检查输出文件是否已存在，如果存在则加载已有数据
    conversation_records = []
    if os.path.exists(output_json_path):
        try:
            with open(output_json_path, 'r', encoding='utf-8') as f:
                conversation_records = json.load(f)
            print(f"已加载现有数据，包含 {len(conversation_records)} 条记录")
        except Exception as e:
            print(f"加载现有数据失败: {e}")
    
    # 获取目录下所有txt文件
    txt_files = [f for f in os.listdir(directory_path) if f.endswith('.txt')]
    
    # 获取已处理文件的ID列表
    processed_ids = set(record["id"] for record in conversation_records)
    
    for i, txt_file in enumerate(txt_files):
        article_path = os.path.join(directory_path, txt_file)
        # 使用MD5哈希算法基于文件名生成稳定的唯一ID
        file_id = hashlib.md5(txt_file.encode('utf-8')).hexdigest()
        
        # 检查是否已处理过该文件
        if file_id in processed_ids:
            print(f"文件 {i+1}/{len(txt_files)}: {txt_file} 已处理，跳过")
            continue
            
        print(f"处理文件 {i+1}/{len(txt_files)}: {txt_file}")
        
        # 获取文章内容
        article_content = read_article_from_file(article_path)
        if article_content == "无法读取文件内容":
            print(f"无法读取文件: {txt_file}, 跳过")
            continue
        
        # 获取总结
        summary = summarize_article(article_path, template_path)
        
        # 过滤掉<think>...</think>内容
        summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
        
        # 创建对话记录
        conversation_record = {
            "id": file_id,
            "conversations": [
                {
                    "from": "user",
                    "value": article_content
                },
                {
                    "from": "assistant",
                    "value": summary
                }
            ]
        }
        
        conversation_records.append(conversation_record)
        
        # 每处理一个文件就保存一次结果
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(conversation_records, f, ensure_ascii=False, indent=2)
        
        print(f"已处理并保存文件 {i+1}/{len(txt_files)}: {txt_file}")
    
    print(f"全部处理完成，共 {len(conversation_records)} 个文件，结果保存到 {output_json_path}")

if __name__ == "__main__":
    # 示例用法
    articles_directory = "/home/mao/workspace/medium_scrape/chunks"
    template_file = "/home/mao/workspace/medium_scrape/flask_endpoint/templates/article_summary_template.txt"
    output_json = "/home/mao/workspace/medium_scrape/flask_endpoint/data_preparation/conversation_records.json"
    
    # 处理单个文件
    # article_file = "/home/mao/workspace/medium_scrape/chunks/1-bit-quantization-run-models-with-trillions-of-parameters-on-your-computer-442617a61440.txt"
    # summary = summarize_article(article_file, template_file)
    # summary = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL)
    # print("\n文章总结:")
    # print(summary)
    
    # 处理整个目录
    process_directory(articles_directory, template_file, output_json)