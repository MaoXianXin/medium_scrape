from web_utils import (
    init_driver,
    create_articles_directory,
    get_article_content,
    save_article,
    remove_url_from_file,
    DEFAULT_ARTICLES_DIR,
    DEFAULT_URLS_FILE,
    DEFAULT_PROCESSED_URLS_FILE
)
import time
import sys
import argparse


"""
python get_urls_content.py --remote-debugging --port 9222
"""
def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='抓取Medium文章内容')
    parser.add_argument('--remote-debugging', action='store_true', help='连接到已运行的Chrome实例')
    parser.add_argument('--port', type=int, default=9222, help='Chrome远程调试端口 (默认: 9222)')
    args = parser.parse_args()
    
    # 配置路径
    chrome_driver_path = "/home/mao/Downloads/chromedriver-linux64/chromedriver"
    chrome_binary_path = "/home/mao/Downloads/chrome-linux64/chrome"
    
    # 配置文件路径
    ARTICLES_DIR = DEFAULT_ARTICLES_DIR  # 使用默认值，也可以自定义
    URLS_FILE = DEFAULT_URLS_FILE  # 使用默认值，也可以自定义
    PROCESSED_URLS_FILE = DEFAULT_PROCESSED_URLS_FILE  # 添加这个配置
    
    # 设置请求间隔时间（秒）- 每分钟6篇文章
    REQUEST_INTERVAL = 10
    
    # 创建文章保存目录
    create_articles_directory(ARTICLES_DIR)
    
    # 初始化浏览器
    if args.remote_debugging:
        driver = init_driver(chrome_driver_path, chrome_binary_path, 
                            use_remote_debugging=True, debugging_port=args.port)
        print(f"已连接到端口 {args.port} 的Chrome实例")
    else:
        driver = init_driver(chrome_driver_path, chrome_binary_path)
        # 等待用户登录并确认
        input("请在浏览器中完成登录，完成后按回车键继续...")
    
    print("开始抓取文章...")
    
    try:
        # 读取文章URL列表
        with open(URLS_FILE, 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        # 读取已处理的URLs
        processed_urls = set()
        try:
            with open(PROCESSED_URLS_FILE, 'r', encoding='utf-8') as f:
                processed_urls = {line.strip() for line in f}
        except FileNotFoundError:
            pass
            
        # 过滤掉已处理的URLs
        urls = [url for url in urls if url not in processed_urls]
        
        # 处理每个URL
        for url in urls:
            article_id = url.split('/')[-1]
            print(f"正在处理文章: {article_id}")
            
            try:
                # 获取并保存文章内容
                article_data = get_article_content(driver, url)
                if save_article(article_data, article_id, ARTICLES_DIR, PROCESSED_URLS_FILE):
                    print(f"文章已保存: {article_id}")
                    remove_url_from_file(url, URLS_FILE)
            except Exception as e:
                print(f"处理文章时出错: {article_id}")
                print(f"错误信息: {str(e)}")
                driver.quit()
                sys.exit(1)
            
            # 使用新的延时间隔
            print(f"等待 {REQUEST_INTERVAL} 秒后处理下一篇文章...")
            time.sleep(REQUEST_INTERVAL)
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()