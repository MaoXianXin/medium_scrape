from difflib import SequenceMatcher

def calculate_match_score(keyword, record):
    """
    计算关键词与记录的匹配得分
    
    参数:
    keyword: 用户输入的单个关键词
    record: 包含三个标签字段的记录
    
    返回:
    float: 匹配得分
    """
    # 权重定义
    weights = {
        'topic_tags': 0.3,
        'technical_tags': 0.4,
        'application_tags': 0.3
    }
    
    final_score = 0
    
    for field, weight in weights.items():
        if field in record and record[field]:
            tags = record[field].split(',')
            
            # 获取匹配标签及分数
            matched_tags, max_tag_score = get_matched_tags(keyword, tags)
            
            # 计算匹配密度
            match_density = calculate_match_density(matched_tags, tags)
            
            # 计算字段匹配得分
            # 1. 最佳匹配标签得分占60%
            best_match_component = max_tag_score * 0.6
            
            # 2. 匹配密度占40%
            density_component = match_density * 0.4
            
            field_score = best_match_component + density_component
            
            # 标签数量归一化
            field_score = normalize_by_tag_count(field_score, len(tags))
            
            final_score += field_score * weight
            
    return final_score

def get_matched_tags(keyword, tags):
    """
    获取匹配的标签及其最高分数
    
    返回:
    list: 匹配的标签列表及对应分数
    float: 最高匹配分数
    """
    matched_tags = []
    max_score = 0
    
    for tag in tags:
        tag = tag.strip()
        score = calculate_tag_similarity(keyword, tag)
        
        if score > 0.2:  # 设定一个最低匹配阈值
            matched_tags.append((tag, score))
            max_score = max(max_score, score)
    
    return matched_tags, max_score

def calculate_tag_similarity(keyword, tag):
    """
    计算关键词与标签的相似度
    """
    # 转换为小写进行比较，避免大小写敏感问题
    keyword_lower = keyword.lower()
    tag_lower = tag.lower()
    
    # 精确匹配
    if keyword_lower == tag_lower:
        return 1.0
        
    # 包含匹配
    if keyword_lower in tag_lower or tag_lower in keyword_lower:
        # 根据长度比例调整得分
        length_ratio = min(len(keyword), len(tag)) / max(len(keyword), len(tag))
        return 0.8 * length_ratio
    
    # 部分词匹配
    if ' ' in tag_lower or '-' in tag_lower:
        words = tag_lower.replace('-', ' ').split()
        for word in words:
            if keyword_lower == word:
                return 0.6
    
    # 简单的编辑距离相似度
    similarity = calculate_similarity(keyword_lower, tag_lower)
    if similarity > 0.6:
        return similarity * 0.5
    
    return 0

def calculate_similarity(s1, s2):
    """
    计算两个字符串的相似度
    """
    # 使用SequenceMatcher计算相似度
    return SequenceMatcher(None, s1, s2).ratio()

def calculate_match_density(matched_tags, total_tags):
    """
    计算匹配密度
    """
    if not total_tags or not matched_tags:
        return 0
    
    # 标签匹配数量/总标签数量
    basic_density = len(matched_tags) / len(total_tags)
    
    # 匹配标签的平均得分
    avg_match_score = sum(score for _, score in matched_tags) / len(matched_tags)
    
    # 综合密度计算
    return (basic_density * 0.5) + (avg_match_score * 0.5)

def normalize_by_tag_count(score, tag_count):
    """
    根据标签数量进行归一化调整
    """
    # 防止标签数量过多的字段获得不公平优势
    if tag_count > 5:
        adjustment_factor = 1 - (tag_count - 5) * 0.03  # 标签数量超过5个时，每多1个标签降低3%的权重
        adjustment_factor = max(0.7, adjustment_factor)  # 最低不低于0.7
        return score * adjustment_factor
    return score


if __name__ == "__main__":
    # 示例记录
    record = {
        'topic_tags': '知识密集型任务,领域适配,可解释性,结构化推理',
        'technical_tags': 'RAG,Fine-Tuning,RAFT,RoG,Knowledge Graph',
        'application_tags': '医疗问答系统,法律案例分析,代码生成任务,AI辅助系统'
    }

    # 查询关键词
    keyword = '任务'

    # 计算匹配得分
    score = calculate_match_score(keyword, record)
    print(f"关键词 '{keyword}' 与记录的匹配得分: {score}")