"""
flowchart TD
    A[用户提供Goal] --> B[LLM接收Goal]
    B --> C[进入处理循环]
    C --> D[从外界获取Input]
    D --> E1{Input是exit?}
    E1 -- 是 --> J[结束循环并显示记忆]
    E1 -- 否 --> E2{Input与Goal相关?}
    E2 -- 否 --> C
    E2 -- 是 --> F[LLM结合Memory、Goal处理Input]
    F --> G[生成新Memory草案]
    G --> H{用户确认更新?}
    H -- 否 --> C
    H -- 是 --> I[更新Memory]
    I --> C
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


class CognitiveArchitecture:
    def __init__(self):
        self.llm = ChatOpenAI(model="deepseek-r1", temperature=0.7, base_url="https://api.gptapi.us/v1", api_key="sk-HuCbzLcW9t2VOc1t49693cFfF5C74f9bB72d179784380cB4")
        self.memory_file = "memory.txt"
        self.current_goal = None
        
        # 初始化时创建或清空记忆文件
        with open(self.memory_file, "w", encoding="utf-8") as f:
            f.write("")

    def _load_memory(self):
        """从文件加载记忆"""
        with open(self.memory_file, "r", encoding="utf-8") as f:
            return f.read()

    def _save_memory(self, content):
        """保存记忆到文件"""
        with open(self.memory_file, "a", encoding="utf-8") as f:
            f.write(f"\n{content}")

    def _get_relevance_judgment(self, input_text):
        """判断输入相关性"""
        prompt = ChatPromptTemplate.from_template(
            "当前目标：{goal}\n输入内容：{input}\n请判断该输入是否与目标相关？只需回答'是'或'否'"
        )
        chain = prompt | self.llm
        response = chain.invoke({
            "goal": self.current_goal,
            "input": input_text
        })
        return "是" in response.content
    
    def _generate_memory_draft(self, input_text):
        """生成记忆草案"""
        prompt = ChatPromptTemplate.from_template(
            "基于当前目标'{goal}'，整合以下信息生成直接支持该目标的记忆要点：\n"
            "历史记忆：{memory}\n"
            "新输入：{input}\n"
            "要求：每个记忆要点必须明确说明与目标的关系"
        )
        chain = prompt | self.llm
        return chain.invoke({
            "goal": self.current_goal,
            "memory": self._load_memory(),
            "input": input_text
        }).content
    
    def _get_user_confirmation(self, draft):
        """获取用户确认"""
        print(f"\n建议更新的记忆内容：\n{draft}")
        return input("是否更新记忆？(y/n): ").lower() == 'y'
    
    def run_cycle(self):
        """主处理循环"""
        while True:
            # 模拟从外部获取输入
            input_text = input("\n请输入新信息（输入'exit'退出）: ")
            
            if input_text.lower() == 'exit':
                break
                
            if not self._get_relevance_judgment(input_text):
                print("→ 输入内容与目标无关，已跳过")
                continue
                
            memory_draft = self._generate_memory_draft(input_text)
            
            if self._get_user_confirmation(memory_draft):
                self._save_memory(memory_draft)
                print("✓ 记忆已更新")
            else:
                print("× 记忆更新已取消")

# 初始化系统
cognitive_system = CognitiveArchitecture()

# 设置初始目标
cognitive_system.current_goal = input("请输入初始目标：")

# 启动处理循环
cognitive_system.run_cycle()

# 查看最终记忆
print("\n最终记忆内容：")
with open(cognitive_system.memory_file, "r", encoding="utf-8") as f:
    print(f.read())
