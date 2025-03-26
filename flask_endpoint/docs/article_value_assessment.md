# 代码逻辑分析：文章价值评估系统

这段代码实现了一个文章价值评估系统，它能够读取文章文件，获取摘要，然后生成详细的价值评估报告。下面是代码的主要逻辑结构：

## 1. 主要组件

### 评估模板
代码开始定义了一个中文的评估模板 `assessment_template`，包含以下几个评估维度：
- 创新性评估 (25分)
- 实用性评估 (25分)
- 完整性评估 (15分)
- 可信度评估 (20分)
- 特定需求评估 (15分)
- 总分评估 (100分)

### LangChain 提示模板
使用 LangChain 创建了一个提示模板 `assessment_prompt_template`，它接收三个输入变量：
- 文章内容 (`article_text`)
- 文章摘要 (`summary`)
- 评估模板 (`assessment_template`)

### 核心处理函数
`process_article()` 函数处理单个文章并生成评估报告，步骤如下：
1. 读取文章文本
2. 通过本地 Flask API 获取文章摘要
3. 使用 LLM 生成文章价值评估
4. 保存评估结果到文件
5. 记录已处理的文章

### 主函数
`main()` 函数是程序入口，它：
1. 解析命令行参数
2. 初始化 LLM (使用 deepseek-r1 模型)
3. 获取输入目录中的所有 txt 文件
4. 加载已处理文章的日志
5. 过滤出未处理的文章
6. 逐个处理未处理的文章

## 2. 数据流向

1. **输入**：文章文本文件 (.txt)
2. **处理**：
   - 读取文章内容
   - 调用本地 Flask API 获取摘要
   - 使用 LLM 生成评估报告
3. **输出**：
   - 评估报告文件 (.md)
   - 已处理文章日志

## 3. 关键技术点

- 使用 LangChain 和 OpenAI API 进行文本生成
- 使用本地 Flask API 进行文章摘要
- 使用正则表达式过滤掉 LLM 输出中的 `<think>...</think>` 内容
- 文件操作和命令行参数解析
- 使用时间戳确保输出文件名唯一性

## 4. 工作流程

1. 用户通过命令行指定输入目录、输出目录和日志文件路径
2. 程序找出所有未处理的文章
3. 对每篇未处理的文章：
   - 读取文章内容
   - 调用 API 获取摘要
   - 使用 LLM 生成评估报告
   - 保存评估报告
   - 更新已处理文章日志
4. 处理完所有文章后程序结束

这个系统设计得比较完整，包含了错误处理、日志记录和进度显示，适合批量处理大量文章并生成结构化的评估报告。


# 文章价值评估系统中的重要参数变量

基于文档内容，以下是代码中需要特别注意的重要参数变量：

## 1. 评估相关参数

- **`assessment_template`**: 定义了评估的维度和分值分配，包括创新性(25分)、实用性(25分)、完整性(15分)、可信度(20分)和特定需求(15分)，总计100分。这是评估标准的核心定义。

## 2. 模型和API相关参数

1. **摘要 API 请求参数**：
   ```python
   payload = {
       "article_text": article_text,
       "temperature": 0.2,
       "model": "deepseek-r1",
       "api_key": "sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
       "base_url": "https://www.gptapi.us/v1"
   }
   ```

2. **LLM 初始化参数**：
   
   ```python
   llm = ChatOpenAI(
       model="deepseek-r1",
       temperature=0.3,
       api_key="sk-GUtb3hUr2MSiUifp343823B7632f412294078bD5A0DcF3C7",
       base_url="https://www.gptapi.us/v1"
   )
   ```

## 3. 命令行参数

- `--input_dir`：包含文章 txt 文件的目录，默认为 `/home/mao/workspace/medium_scrape/111`
- `--output_dir`：保存文章价值评估的目录，默认为 `/home/mao/workspace/medium_scrape/assessments`
- `--log_path`：已处理文章的日志文件路径，默认为 `/home/mao/workspace/medium_scrape/flask_endpoint/processed_articles.txt`

## 4. 文件处理参数

- **文件扩展名过滤**: 代码专门处理`.txt`文件作为输入。
- **输出文件命名**: 使用时间戳确保文件名唯一性。

## 5. 提示模板参数

- **`assessment_prompt_template`**: LangChain提示模板，接收三个关键输入变量:
  - `article_text`: 完整的文章内容
  - `summary`: 通过API获取的文章摘要
  - `assessment_template`: 评估标准模板

## 6. 正则表达式参数

- **思考内容过滤正则**: 用于过滤掉LLM输出中的`<think>...</think>`内容的正则表达式。

## 7. 处理控制参数

- **已处理文章列表**: 用于跟踪哪些文章已经被处理，避免重复处理。
- **错误处理超时设置**: 可能存在API调用或LLM生成的超时设置。

这些参数对系统的运行至关重要，在使用或修改代码时需要特别注意它们的设置和影响。如果需要调整系统的评估标准、输出格式或处理流程，这些参数将是主要的调整点。
