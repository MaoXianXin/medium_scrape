from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from web_utils import init_driver
import os
import time
import sys

def create_articles_directory():
    """创建保存文章的目录"""
    if not os.path.exists('articles'):
        os.makedirs('articles')

def get_article_content(driver, url):
    """获取文章内容"""
    try:
        driver.get(url)
        # 等待文章主体加载完成
        article = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, "article"))
        )
        
        # 获取文章标题
        title = driver.find_element(By.TAG_NAME, "h1").text
        
        # 获取所有段落内容
        paragraphs = article.find_elements(By.TAG_NAME, "p")
        content = [p.text for p in paragraphs if p.text.strip()]
        
        return {
            'title': title,
            'content': '\n\n'.join(content)
        }
    except Exception as e:
        print(f"获取文章内容时出错: {url}")
        print(f"错误信息: {str(e)}")
        driver.quit()  # 确保关闭浏览器
        sys.exit(1)    # 终止程序运行

def save_article(article_data, article_id):
    """保存文章内容到文件"""
    if not article_data:
        return False
    
    filepath = os.path.join('articles', f'{article_id}.txt')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"标题：{article_data['title']}\n\n")
            f.write(article_data['content'])
        return True
    except Exception as e:
        print(f"保存文章时出错: {article_id}")
        print(f"错误信息: {str(e)}")
        return False

def remove_url_from_file(url):
    """从文件中移除已处理的URL"""
    try:
        with open('article_urls.txt', 'r', encoding='utf-8') as f:
            urls = f.readlines()
        
        with open('article_urls.txt', 'w', encoding='utf-8') as f:
            for line in urls:
                if line.strip() != url.strip():
                    f.write(line)
    except Exception as e:
        print(f"移除URL时出错: {url}")
        print(f"错误信息: {str(e)}")

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
            
            # 获取并保存文章内容
            article_data = get_article_content(driver, url)
            if save_article(article_data, article_id):
                print(f"文章已保存: {article_id}")
                remove_url_from_file(url)  # 移除已处理的URL
            
            # 添加延时，避免请求过于频繁
            time.sleep(2)
            
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        driver.quit()

if __name__ == "__main__":
    main()