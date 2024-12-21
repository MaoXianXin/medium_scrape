from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import time
import os
import re
import random
from tqdm import tqdm

# 设置 ChromeDriver 的路径
chrome_driver_path = "/home/mao/Downloads/chromedriver-linux64/chromedriver"  # 替换为你的 chromedriver 路径

# 设置 Chrome AppImage 文件路径
chrome_appimage_path = "/home/mao/Downloads/chrome-linux64/chrome"  # 替换为你的 Chrome AppImage 文件路径

# 配置 Chrome Options
chrome_options = Options()
chrome_options.binary_location = chrome_appimage_path  # 设置 Chrome 的二进制文件路径

# 启动 ChromeDriver 服务
service = Service(chrome_driver_path)

# 启动浏览器
driver = webdriver.Chrome(service=service, options=chrome_options)

# 创建保存文章的目录
save_dir = "articles"  # 可以修改为你想要的目录名
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 添加用于保存和加载URL的函数
def save_urls_to_file(urls, filename='pending_urls.txt'):
    with open(filename, 'w') as f:
        # 每行写入一个URL，末尾加换行符
        for url in urls:
            f.write(url + '\n')

def load_urls_from_file(filename='pending_urls.txt'):
    try:
        with open(filename, 'r') as f:
            # 读取所有行并去除每行末尾的空白字符
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

# 添加新函数用于保存非文章链接
def save_non_article_urls(urls, filename='non_article_urls.txt'):
    with open(filename, 'w') as f:
        for url in urls:
            f.write(url + '\n')

def load_non_article_urls(filename='non_article_urls.txt'):
    try:
        with open(filename, 'r') as f:
            return set(line.strip() for line in f)
    except FileNotFoundError:
        return set()

# 添加函数用于判断URL是否为文章链接
def is_article_url(url):
    # 排除明显的非文章URL
    non_article_patterns = [
        '/about$',                          # 个人简介页面，以 /about 结尾
        '/following$',                      # 关注列表页面
        '/followers$',                      # 粉丝列表页面
        'signin',                          # 登录页面
        'privacy-policy',                  # 隐私政策页面
        'terms-of-service',                # 服务条款页面
        'help.medium.com',                 # Medium 帮助中心页面
        'statuspage.io',                   # Medium 服务状态页面
        '^https://medium.com/@[^/]+$',     # 用户主页 URL
        'pressinquiries@medium.com',       # Medium 新闻咨询邮箱
        '/data-science-at-microsoft$',     # Microsoft 数据科学专题页
        'speechify.com',                   # Speechify 网站链接
        '/business$',                      # Medium 商业版块主页
        '/towards-data-science$',          # Towards Data Science 专题主页
        'blog.medium.com',                 # Medium 官方博客
        'jobs-at-medium',                  # Medium 招聘页面
        'work-at-medium',                  # Medium 工作机会页面
        '/list/',                          # Medium 列表页面
        '/lists$',                         # 用户的列表集合页面
        '/ai-in-plain-english$',           # 专题主页
    ]
    
    for pattern in non_article_patterns:
        if re.search(pattern, url):
            return False
    return True

try:
    # 加载之前未处理的URLs和非文章URLs
    pending_urls = load_urls_from_file()
    non_article_urls = load_non_article_urls()
    
    # 过滤pending_urls中的非文章链接
    article_urls = set()
    new_non_article_urls = set()
    
    for url in pending_urls:
        if is_article_url(url):
            article_urls.add(url)
        else:
            new_non_article_urls.add(url)
    
    # 更新非文章URLs并保存
    non_article_urls.update(new_non_article_urls)
    save_non_article_urls(non_article_urls)
    
    # 更新待处理的文章URLs
    pending_urls = article_urls
    save_urls_to_file(pending_urls)
    
    print(f"\n待处理文章数量: {len(pending_urls)}")
    print(f"已识别非文章链接数量: {len(non_article_urls)}")
    
    # 如果没有待处理的URLs，则重新抓取
    if not pending_urls:
        # 打开博主主页
        base_url = "https://medium.com/@zaiinn440"
        driver.get(base_url)

        # 暂停执行,给用户时间检查元素
        print("请打开开发者工具(F12),使用元素选择器(Ctrl+Shift+C)来查找文章链接的CSS selector")
        print("确认后按Enter继续...")
        input()

        # 这里可以根据实际检查到的selector修改选择器
        article_elements = driver.find_elements(By.CSS_SELECTOR, "a[rel='noopener follow']")
        
        # 提取并构建完整的文章URL
        article_urls = set()  # 使用 set 来自动去重
        for element in article_elements:
            href = element.get_attribute('href')
            if href and 'source=user_profile' in href:
                # 提取文章的基础URL（移除参数）
                base_url = href.split('?')[0]
                article_urls.add(base_url)
        
        # 打印所有文章URL（现在是去重后的）
        print(f"\n找到 {len(article_urls)} 篇文章:")
        for url in article_urls:
            print(url)
        
        pending_urls = article_urls
        # 保存所有URLs
        save_urls_to_file(pending_urls)
    
    print(f"\n待处理文章数量: {len(pending_urls)}")
    
    # 创建已处理URL集合
    processed_urls = set()
    
    # 抓取所有文章的内容
    # Convert pending_urls to a list for iteration
    urls_to_process = list(pending_urls)
    for article_url in tqdm(urls_to_process, desc="抓取进度"):
        try:
            # 构建文件名（基于URL）
            url_title = article_url.split('/')[-1]  # 从URL获取最后一部分作为标题
            safe_title = re.sub(r'[^\w\s-]', '_', url_title)
            filename = os.path.join(save_dir, f"{safe_title}.txt")
            
            # 检查文件是否已存在
            if os.path.exists(filename):
                tqdm.write(f"文章已存在，跳过: {article_url}")
                # 从pending_urls中移除已存在的文章URL
                pending_urls.remove(article_url)
                # 保存更新后的待处理URLs
                save_urls_to_file(pending_urls)
                continue
            
            # 添加随机延时（2-5秒）
            delay = random.uniform(2, 5)
            tqdm.write(f"等待 {delay:.1f} 秒...")
            time.sleep(delay)
            
            tqdm.write(f"\n正在抓取文章: {article_url}")
            
            # 访问文章页面
            driver.get(article_url)
            
            # 等待页面加载（最多等待20秒）
            wait = WebDriverWait(driver, 20)
            
            # 等待并获取文章标题
            title = wait.until(EC.presence_of_element_located((By.TAG_NAME, "h1"))).text
            
            # 获取文章内容
            try:
                article_content = wait.until(EC.presence_of_element_located((By.TAG_NAME, "article"))).text
            except (TimeoutException, NoSuchElementException):
                print(f"无法获取文章内容: {article_url}")
                continue
            
            # 保存文章内容
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"标题：{title}\n\n")
                f.write(f"URL：{article_url}\n\n")
                f.write("正文：\n")
                f.write(article_content)
            
            tqdm.write(f"文章已保存至: {filename}")
            
            # 文章处理成功后，添加到已处理集合
            processed_urls.add(article_url)
            
            # 更新待处理URLs并保存
            pending_urls.remove(article_url)
            save_urls_to_file(pending_urls)
            
        except Exception as e:
            tqdm.write(f"处理文章时出错 {article_url}: {str(e)}")
            # 保存当前待处理的URLs
            save_urls_to_file(pending_urls)
            continue

finally:
    # 保存最终的待处理URLs
    save_urls_to_file(pending_urls)
    # 关闭浏览器
    driver.quit()
