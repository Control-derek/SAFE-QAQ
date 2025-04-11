import json
import re
import argparse
import os
import logging
from datetime import datetime

def setup_logger(log_file):
    
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

def extract_scene_classification(response, field_types):
    
    json_pattern = r'\{[\s\S]*?"scene"[\s\S]*?\}'
    json_match = re.search(json_pattern, response)
    
    if json_match:
        try:
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            
            scene = data.get("scene", "未知")
            
            
            if scene not in field_types:
                for field_type in field_types:
                    if field_type in scene:
                        scene = field_type
                        break
                else:
                    scene = "未知"
                    
            return scene
        except json.JSONDecodeError:
            pass
    
    
    scene_pattern = r'"scene"\s*:\s*"([^"]+)"'
    scene_match = re.search(scene_pattern, response)
    scene = scene_match.group(1) if scene_match else "未知"
    
    
    if scene not in field_types:
        for field_type in field_types:
            if field_type in scene:
                scene = field_type
                break
        else:
            scene = "未知"
    
    return scene

def extract_fraud_classification(response):
    
    json_pattern = r'\{[\s\S]*?"is_fraud"[\s\S]*?\}'
    json_match = re.search(json_pattern, response)
    
    if json_match:
        try:
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            
            is_fraud = data.get("is_fraud", False)
            
            
            if isinstance(is_fraud, str):
                is_fraud = is_fraud.lower() == "true"
                
            return "fraud" if is_fraud else "normal"
        except json.JSONDecodeError:
            pass
    
    
    is_fraud_pattern = r'"is_fraud"\s*:\s*(true|false)'
    is_fraud_match = re.search(is_fraud_pattern, response, re.IGNORECASE)
    is_fraud = is_fraud_match.group(1).lower() == "true" if is_fraud_match else False
    
    
    if not is_fraud_match:
        if re.search(r"(是诈骗|属于诈骗|判断为诈骗|欺诈电话|欺诈行为|确定是诈骗)", response):
            is_fraud = True
        elif re.search(r"(不是诈骗|非诈骗|正常|合法|不属于诈骗|判断为正常)", response):
            is_fraud = False
    
    return "fraud" if is_fraud else "normal"

def extract_fraud_type_classification(response, fraud_types):
    
    json_pattern = r'\{[\s\S]*?"fraud_type"[\s\S]*?\}'
    json_match = re.search(json_pattern, response)
    
    if json_match:
        try:
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            
            fraud_type = data.get("fraud_type", "未知")
            
            
            if fraud_type not in fraud_types:
                for f_type in fraud_types:
                    if f_type in fraud_type:
                        fraud_type = f_type
                        break
                else:
                    fraud_type = "未知"
                    
            return fraud_type
        except json.JSONDecodeError:
            pass
    
    
    fraud_type_pattern = r'"fraud_type"\s*:\s*"([^"]+)"'
    fraud_type_match = re.search(fraud_type_pattern, response)
    fraud_type = fraud_type_match.group(1) if fraud_type_match else "未知"
    
    
    if fraud_type not in fraud_types:
        for f_type in fraud_types:
            if f_type in fraud_type:
                fraud_type = f_type
                break
        else:
            fraud_type = "未知"
    
    return fraud_type

def find_best_responses(input_file, output_file, incorrect_file, log_file, verbose=False):
    
    logger = setup_logger(log_file)
    
    
    logger.info("=" * 80)
    logger.info(f"脚本开始执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"输入文件: {input_file}")
    logger.info(f"输出文件: {output_file}")
    logger.info(f"错误提示文件: {incorrect_file}")
    logger.info(f"日志文件: {log_file}")
    logger.info("=" * 80)
    
    
    FIELD_TYPES = [
        "订餐服务", "咨询客服", "预约服务", 
        "交通咨询", "日常购物", 
        "打车服务", "外卖服务"
    ]
    
    FRAUD_TYPES = [
        "投资诈骗", "钓鱼诈骗", "身份盗窃", 
        "彩票诈骗", "银行诈骗", "绑架诈骗", 
        "客服诈骗", "邮件诈骗"
    ]
    
    logger.info(f"正在加载输入文件: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"加载了 {len(data)} 个样本")
    except Exception as e:
        logger.error(f"加载输入文件失败: {str(e)}")
        return
    
    
    prompt_groups = {}
    for item in data:
        prompt_key = json.dumps(item["prompt"])  
        if prompt_key not in prompt_groups:
            prompt_groups[prompt_key] = []
        prompt_groups[prompt_key].append(item)
    
    logger.info(f"识别出 {len(prompt_groups)} 个独特提示")
    
    
    best_responses = []
    incorrect_prompts = []
    task_stats = {"场景分类": 0, "欺诈分类": 0, "欺诈类型分类": 0, "未知任务": 0}
    correct_stats = {"场景分类": 0, "欺诈分类": 0, "欺诈类型分类": 0, "未知任务": 0}
    
    logger.info("\n每个提示的响应长度信息:")
    prompt_count = 0
    
    for prompt_key, samples in prompt_groups.items():
        prompt_count += 1
        
        prompt = json.loads(prompt_key)
        prompt_length = len(prompt)
        
        
        task_type = "未知任务"
        if prompt_length == 2:
            task_type = "场景分类"
        elif prompt_length == 4:
            task_type = "欺诈分类"
        elif prompt_length == 6:
            task_type = "欺诈类型分类"
        
        task_stats[task_type] += 1
        
        correct_samples = []
        expected_answer = samples[0]["answer"]  
        
        
        longest_response = max(samples, key=lambda x: len(x["generated"]))
        longest_length = len(longest_response["generated"])
        
        
        for sample in samples:
            response = sample["generated"]
            
            
            if task_type == "场景分类":
                prediction = extract_scene_classification(response, FIELD_TYPES)
            elif task_type == "欺诈分类":
                prediction = extract_fraud_classification(response)
            elif task_type == "欺诈类型分类":
                prediction = extract_fraud_type_classification(response, FRAUD_TYPES)
            else:
                
                prediction = None
            
            
            if prediction == expected_answer:
                correct_samples.append(sample)
        
        
        if correct_samples:
            best_response = min(correct_samples, key=lambda x: len(x["generated"]))
            correct_stats[task_type] += 1
            
            logger.info(f"  - 正确响应数量: {len(correct_samples)}")
            logger.info(f"  - 选择的响应长度: {len(best_response['generated'])} 个字符 (最短的正确响应)")
            
            best_responses.append(best_response)
        else:
            
            shortest_incorrect = min(samples, key=lambda x: len(x["generated"]))
            
            logger.info(f"  - 正确响应数量: 0")
            logger.info(f"  - 选择的响应长度: {len(shortest_incorrect['generated'])} 个字符 (最短的响应，但不正确)")
            
            
            incorrect_prompts.append({
                "task_type": task_type,
                "prompt": prompt,
                "expected_answer": expected_answer,
                "best_incorrect_response": shortest_incorrect["generated"],
                "all_responses": [sample["generated"] for sample in samples]
            })
    
    
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(best_responses, f, ensure_ascii=False, indent=2)
        logger.info(f"\n选择了 {len(best_responses)} 个最佳正确响应，已保存到: {output_file}")
    except Exception as e:
        logger.error(f"保存输出文件失败: {str(e)}")
    
    
    if incorrect_prompts:
        try:
            with open(incorrect_file, 'w', encoding='utf-8') as f:
                json.dump(incorrect_prompts, f, ensure_ascii=False, indent=2)
            logger.info(f"\n发现 {len(incorrect_prompts)} 个没有正确响应的提示，已保存到: {incorrect_file}")
        except Exception as e:
            logger.error(f"保存错误提示文件失败: {str(e)}")
    else:
        logger.info("\n所有提示都有至少一个正确响应！")
    
    
    logger.info("\n任务类型统计:")
    for task, count in task_stats.items():
        correct = correct_stats[task]
        if count > 0:
            correct_rate = correct / count * 100
            logger.info(f"  - {task}: 共 {count} 个提示，找到 {correct} 个正确响应 (正确率: {correct_rate:.2f}%)")
    
    logger.info(f"\n共处理了 {len(prompt_groups)} 个独特提示")
    logger.info(f"选择了 {len(best_responses)} 个最佳正确响应")
    logger.info(f"输出已保存到 {output_file}")
    
    
    logger.info("=" * 80)
    logger.info(f"脚本结束执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)

def main():
    
    parser = argparse.ArgumentParser(description="选择每个提示的最佳响应（正确且最短）")
    
    parser.add_argument(
        "--input", "-i", 
        required=True, 
        help="输入JSON文件路径，包含多个提示及其响应"
    )
    
    parser.add_argument(
        "--output", "-o", 
        required=True, 
        help="输出JSON文件路径，将保存每个提示的最佳正确响应"
    )
    
    parser.add_argument(
        "--incorrect", "-ic", 
        default=None, 
        help="保存没有正确响应的提示的文件路径，默认为'incorrect_prompts.json'"
    )
    
    parser.add_argument(
        "--log", "-l", 
        default=None, 
        help="日志文件路径，默认为'script_log_日期时间.log'"
    )
    
    parser.add_argument(
        "--verbose", "-v", 
        action="store_true", 
        help="显示详细处理信息"
    )
    
    args = parser.parse_args()
    
    
    if args.incorrect is None:
        
        output_dir = os.path.dirname(args.output) or "."
        output_filename = os.path.basename(args.output)
        output_name, output_ext = os.path.splitext(output_filename)
        
        
        args.incorrect = os.path.join(output_dir, f"{output_name}_incorrect{output_ext}")
    
    
    if args.log is None:
        
        output_dir = os.path.dirname(args.output) or "."
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        
        args.log = os.path.join(output_dir, f"script_log_{timestamp}.log")
    
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.incorrect)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.log)), exist_ok=True)
    
    
    find_best_responses(args.input, args.output, args.incorrect, args.log, args.verbose)

if __name__ == "__main__":
    main()