from utils import OpenAIClient
from vector_utils import VectorSearcher
import argparse
from pathlib import Path

"""
向量搜索工具 - 用于构建和搜索文档的向量索引

使用方法:
1. 普通运行（自动处理首次构建或增量更新）:
   python article_filter_vector.py --query "注意力机制"

2. 强制重建索引:
   python article_filter_vector.py --force-rebuild --query "注意力机制"

3. 指定返回结果数量:
   python article_filter_vector.py --query "注意力机制" --top-k 3

4. 仅更新索引，不执行搜索:
   python article_filter_vector.py

参数说明:
--force-rebuild: 强制重建索引
--query: 搜索查询词
--top-k: 返回结果数量（默认为5）
"""

def initialize_searcher(api_key: str, base_url: str, model_name: str) -> VectorSearcher:
    """初始化向量搜索器"""
    ai_client = OpenAIClient(api_key, base_url, model_name)
    return VectorSearcher(ai_client)

def setup_vector_index(searcher: VectorSearcher, summaries_dir: str, force_rebuild: bool = False):
    """设置向量索引，处理首次构建和增量更新"""
    try:
        # 检查索引文件是否存在
        vectors_exist = Path(searcher.vectors_file).exists()
        metadata_exist = Path(searcher.metadata_file).exists()
        
        if not vectors_exist or not metadata_exist:
            # 首次运行，构建完整索引
            print("未找到现有索引，开始构建完整索引...")
            searcher.build_index(summaries_dir, force_rebuild=True)
        else:
            if force_rebuild:
                # 强制重建索引
                print("强制重建索引...")
                searcher.build_index(summaries_dir, force_rebuild=True)
            else:
                # 增量更新
                print("检查索引更新...")
                searcher.update_index(summaries_dir)
                
    except Exception as e:
        print(f"索引构建/更新失败: {str(e)}")
        raise

def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description='向量搜索工具')
    parser.add_argument('--force-rebuild', action='store_true', 
                       help='强制重建索引')
    parser.add_argument('--query', type=str, default=None,
                       help='搜索查询词')
    parser.add_argument('--top-k', type=int, default=5,
                       help='返回结果数量')
    args = parser.parse_args()

    # 配置参数
    api_key = "sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4"
    base_url = "https://www.gptapi.us/v1"
    model_name = "gpt-4o-mini"
    summaries_dir = "./summaries"

    try:
        # 初始化搜索器
        searcher = initialize_searcher(api_key, base_url, model_name)
        
        # 设置索引
        setup_vector_index(searcher, summaries_dir, args.force_rebuild)
        
        # 如果提供了查询词，执行搜索
        if args.query:
            results = searcher.search(args.query, top_k=args.top_k)
            print(f"\n搜索结果 - 查询词: '{args.query}'")
            print("-" * 50)
            for i, result in enumerate(results, 1):
                print(f"{i}. 文件名: {result['filename']}")
                print(f"   相似度: {result['similarity_score']:.4f}")
                print()
                
    except Exception as e:
        print(f"程序执行出错: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())