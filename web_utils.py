from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import os

def init_driver(chrome_driver_path, chrome_binary_path):
    """初始化并返回配置好的Chrome WebDriver"""
    chrome_options = Options()
    chrome_options.binary_location = chrome_binary_path
    service = Service(chrome_driver_path)
    return webdriver.Chrome(service=service, options=chrome_options)

def save_urls_to_file(urls, filename='article_urls.txt'):
    """将URL列表保存到文件"""
    # 获取已存在的文章
    existing_articles = set()
    try:
        for file in os.listdir('articles'):
            name, ext = os.path.splitext(file)
            if ext.lower() == '.txt':
                existing_articles.add(name)
    except FileNotFoundError:
        pass

    # 过滤出新的URL
    new_urls = []
    for url in urls:
        article_id = url.split('/')[-1]  # 获取URL最后一部分作为文章ID
        if article_id not in existing_articles:
            new_urls.append(url)

    # 保存新的URL
    with open(filename, 'w', encoding='utf-8') as f:
        for url in new_urls:
            f.write(url + '\n')
    
    return len(urls) - len(new_urls)  # 返回跳过的URL数量

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
            
        # 2. 跳过出版物主页 (URL结构为 medium.com/publication-name)
        url_parts = base_url.split('/')
        if len(url_parts) <= 4:  # https://medium.com/publication-name
            should_skip = True
            
        # 3. 跳过用户主页 (URL结构为 medium.com/@username)
        if len(url_parts) == 4 and url_parts[3].startswith('@'):
            should_skip = True
            
        if not should_skip:
            article_urls.add(base_url)
    
    return article_urls