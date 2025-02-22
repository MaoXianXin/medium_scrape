import tiktoken
import os

def count_tokens(file_path, model="gpt-4-turbo", save_cleaned=True):
    """
    计算指定文件的tokens数量，去除文件名和路径信息
    
    Args:
        file_path (str): 文件路径
        model (str, optional): 使用的模型名称，默认为"gpt-4-turbo"
        save_cleaned (bool, optional): 是否保存清理后的内容，默认为True
        
    Returns:
        tuple: (original_token_count, cleaned_token_count, cleaned_file_path) 如果save_cleaned为True
              (original_token_count, cleaned_token_count) 如果save_cleaned为False
    """
    try:
        # 初始化tokenizer
        tokenizer = tiktoken.encoding_for_model(model)
        
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # 计算原始文本的tokens数量
        original_tokens = tokenizer.encode(text, disallowed_special=())
        original_token_count = len(original_tokens)
        
        # 去除包含 === 的行以及文件名和路径信息
        cleaned_lines = []
        skip_next_line = False
        last_line_empty = False  # 追踪上一行是否为空行
        
        for line in text.split('\n'):
            if '=' * 10 in line:
                skip_next_line = True
                continue
            if skip_next_line:
                if line.startswith('文件名：') or line.startswith('文件路径：'):
                    continue
                skip_next_line = False
            
            # 处理空行
            if not line.strip():
                if not last_line_empty:  # 只有当上一行不是空行时才添加
                    cleaned_lines.append(line)
                    last_line_empty = True
            else:
                cleaned_lines.append(line)
                last_line_empty = False
            
        cleaned_text = '\n'.join(cleaned_lines)
        
        # 保存清理后的内容
        if save_cleaned:
            # 生成新文件路径
            dir_path = os.path.dirname(file_path)
            file_name = os.path.basename(file_path)
            cleaned_file_path = os.path.join(dir_path, 'cleaned_' + file_name)
            
            # 写入新文件
            with open(cleaned_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_text)
            
        # 计算清理后文本的tokens数量
        tokens = tokenizer.encode(cleaned_text, disallowed_special=())
        cleaned_token_count = len(tokens)
        
        if save_cleaned:
            return (original_token_count, cleaned_token_count, cleaned_file_path)
        return (original_token_count, cleaned_token_count)
        
    except Exception as e:
        print(f"处理文件时发生错误: {str(e)}")
        return None

# 使用示例
if __name__ == "__main__":
    file_path = "/home/mao/workspace/notebook/medium_scrape/summary_results/all_key_points.txt"  # 替换为你的文件路径
    result = count_tokens(file_path)
    if result is not None:
        if len(result) == 3:
            orig_count, cleaned_count, cleaned_file_path = result
            print(f"原始文件tokens数量: {orig_count}")
            print(f"清理后tokens数量: {cleaned_count}")
            print(f"减少的tokens数量: {orig_count - cleaned_count}")
            print(f"清理后的内容已保存至: {cleaned_file_path}")
        else:
            orig_count, cleaned_count = result
            print(f"原始文件tokens数量: {orig_count}")
            print(f"清理后tokens数量: {cleaned_count}")
            print(f"减少的tokens数量: {orig_count - cleaned_count}")