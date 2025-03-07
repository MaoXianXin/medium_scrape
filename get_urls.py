from web_utils import (
    init_driver, 
    save_urls_to_file, 
    extract_article_urls,
    DEFAULT_URLS_FILE,
    DEFAULT_ARTICLES_DIR
)

# 设置路径
chrome_driver_path = "/home/mao/Downloads/chromedriver-linux64/chromedriver"
chrome_appimage_path = "/home/mao/Downloads/chrome-linux64/chrome"

# 配置文件路径
ARTICLES_DIR = DEFAULT_ARTICLES_DIR  # 使用默认值，也可以自定义
URLS_FILE = DEFAULT_URLS_FILE  # 使用默认值，也可以自定义

def main():
    driver = init_driver(chrome_driver_path, chrome_appimage_path)
    
    try:
        # 获取用户输入并打开页面
        base_url = input("请输入博主主页链接: ")
        driver.get(base_url)

        # 等待用户确认
        print("请打开开发者工具(F12),使用元素选择器(Ctrl+Shift+C)来查找文章链接的CSS selector")
        print("确认后按Enter继续...")
        input()

        # 提取文章URL
        article_urls = extract_article_urls(driver)
        
        # 显示结果
        print(f"\n找到 {len(article_urls)} 篇文章:")
        for url in article_urls:
            print(url)

        # 保存结果
        skipped_count = save_urls_to_file(article_urls, URLS_FILE, ARTICLES_DIR)
        print(f"\n找到 {len(article_urls)} 篇文章，其中 {skipped_count} 篇已存在")
        print(f"新文章链接已保存到 {URLS_FILE}")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()