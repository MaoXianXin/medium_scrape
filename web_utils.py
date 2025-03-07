from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import os

# 添加默认配置
DEFAULT_ARTICLES_DIR = 'articles'
DEFAULT_URLS_FILE = 'article_urls.txt'
DEFAULT_PROCESSED_URLS_FILE = 'processed_urls.txt'  # 新增：已处理URL的记录文件

def init_driver(chrome_driver_path, chrome_binary_path):
    """初始化并返回配置好的Chrome WebDriver"""
    chrome_options = Options()
    chrome_options.binary_location = chrome_binary_path
    service = Service(chrome_driver_path)
    return webdriver.Chrome(service=service, options=chrome_options)

def save_urls_to_file(urls, filename=DEFAULT_URLS_FILE, processed_file=DEFAULT_PROCESSED_URLS_FILE):
    """将URL列表保存到文件，并跳过已处理的URL"""
    # 读取已处理的URL
    processed_urls = set()
    try:
        with open(processed_file, 'r', encoding='utf-8') as f:
            for line in f:
                processed_urls.add(line.strip())
    except FileNotFoundError:
        # 如果文件不存在，创建一个空集合
        pass

    # 过滤出新的URL
    new_urls = []
    for url in urls:
        if url not in processed_urls:
            new_urls.append(url)

    # 保存新的URL
    with open(filename, 'w', encoding='utf-8') as f:
        for url in new_urls:
            f.write(url + '\n')
    
    return len(urls) - len(new_urls)  # 返回跳过的URL数量

def add_to_processed_urls(url, processed_file=DEFAULT_PROCESSED_URLS_FILE):
    """将URL添加到已处理列表中"""
    with open(processed_file, 'a', encoding='utf-8') as f:
        f.write(url + '\n')

def extract_article_urls(driver, selector="a[rel='noopener follow']"):
    """提取页面中的文章URL"""
    article_elements = driver.find_elements(By.CSS_SELECTOR, selector)
    article_urls = set()
    
    # 定义非文章URL的模式
    non_article_patterns = [
        '/about',
        '/followers',
        '/lists',
        '/signin',
        'm/signin',
    ]
    
    for element in article_elements:
        href = element.get_attribute('href')
        if not href:
            continue
            
        # 移除URL中的查询参数
        base_url = href.split('?')[0]
        
        # 跳过非文章URL
        should_skip = False
        
        # 1. 跳过包含特定路径的URL
        if any(pattern in base_url for pattern in non_article_patterns):
            should_skip = True
            
        if not should_skip:
            article_urls.add(base_url)
    
    return article_urls

def create_articles_directory(articles_dir=DEFAULT_ARTICLES_DIR):
    """创建保存文章的目录"""
    if not os.path.exists(articles_dir):
        os.makedirs(articles_dir)

def get_article_content(driver, url):
    """获取文章内容"""
    try:
        driver.get(url)
        # 等待页面加载（最多等待20秒）
        wait = WebDriverWait(driver, 20)
        
        # 等待并获取文章标题
        title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1"))).text
        
        # 获取文章内容
        article_content = wait.until(EC.presence_of_element_located((By.TAG_NAME, "article"))).text
        
        return {
            'title': title,
            'url': url,
            'content': article_content
        }
    except Exception as e:
        raise  # 让调用者处理异常

def save_article(article_data, article_id, articles_dir=DEFAULT_ARTICLES_DIR, processed_file=DEFAULT_PROCESSED_URLS_FILE):
    """保存文章内容到文件，并记录已处理的URL"""
    if not article_data:
        return False
    
    filepath = os.path.join(articles_dir, f'{article_id}.txt')
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(f"标题：{article_data['title']}\n\n")
            f.write(f"URL：{article_data['url']}\n\n")  # 添加URL
            f.write("正文：\n")  # 添加正文标识
            f.write(article_data['content'])
        
        # 将URL添加到已处理列表
        add_to_processed_urls(article_data['url'], processed_file)
        return True
    except Exception as e:
        print(f"保存文章时出错: {article_id}")
        print(f"错误信息: {str(e)}")
        return False

def remove_url_from_file(url, urls_file=DEFAULT_URLS_FILE):
    """从文件中移除已处理的URL"""
    try:
        with open(urls_file, 'r', encoding='utf-8') as f:
            urls = f.readlines()
        
        with open(urls_file, 'w', encoding='utf-8') as f:
            for line in urls:
                if line.strip() != url.strip():
                    f.write(line)
    except Exception as e:
        print(f"移除URL时出错: {url}")
        print(f"错误信息: {str(e)}")