from web_utils import (
    init_driver,
    create_articles_directory,
    get_article_content,
    save_article,
    remove_url_from_file
)
import time
import sys

def main():
    # 配置路径
    chrome_driver_path = "/home/mao/Downloads/chromedriver-linux64/chromedriver"
    chrome_binary_path = "/home/mao/Downloads/chrome-linux64/chrome"
    
    # 创建文章保存目录
    create_articles_directory()
    
    # 初始化浏览器
    driver = init_driver(chrome_driver_path, chrome_binary_path)
    
    try:
        # 读取文章URL列表
        with open('article_urls.txt', 'r', encoding='utf-8') as f:
            urls = [line.strip() for line in f if line.strip()]
        
        # 处理每个URL
        for url in urls:
            article_id = url.split('/')[-1]
            print(f"正在处理文章: {article_id}")
            
            try:
                # 获取并保存文章内容
                article_data = get_article_content(driver, url)
                if save_article(article_data, article_id):
                    print(f"文章已保存: {article_id}")
                    remove_url_from_file(url)  # 移除已处理的URL
            except Exception as e:
                print(f"处理文章时出错: {article_id}")
                print(f"错误信息: {str(e)}")
                driver.quit()
                sys.exit(1)
            
            # 添加延时，避免请求过于频繁
            time.sleep(5)
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()