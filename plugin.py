import re
import json
import math
import os
import datetime

from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


class PromptFormatORM(ORM):
    def __call__(self, completions, **kwargs) -> list[float]:
        import re
        import json
        
        basic_pattern = r"<think>[\s\S]*?</think>\s*<answer>[\s\S]*?</answer>"
        
        rewards = []
        for content in completions:
            reward = 0.0
            
            if re.search(basic_pattern, content, re.DOTALL):
                
                reward += 0.5
                try:
                    
                    answer_match = re.search(r"<answer>\s*([\s\S]*?)\s*</answer>", content, re.DOTALL)
                    if answer_match:
                        answer_content = answer_match.group(1).strip()

                        
                        code_block_json = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", answer_content, re.DOTALL)
                        if code_block_json:
                            json_str = code_block_json.group(1)
                        else:
                            
                            json_match = re.search(r"(\{[\s\S]*?\})", answer_content, re.DOTALL)
                            json_str = json_match.group(1) if json_match else None
                        
                        if json_str:
                            try:
                                json_obj = json.loads(json_str)
                                
                                
                                if self._validate_scene_classification(json_obj) or self._validate_fraud_classification(json_obj) or self._validate_fraud_type_classification(json_obj):
                                    reward += 0.5  
                            except json.JSONDecodeError:
                                
                                cleaned_json = re.sub(r'[\n\r\t]', ' ', json_str)
                                try:
                                    json_obj = json.loads(cleaned_json)
                                    
                                    
                                    if self._validate_scene_classification(json_obj) or self._validate_fraud_classification(json_obj) or self._validate_fraud_type_classification(json_obj):
                                        reward += 0.5
                                except:
                                    pass
                except Exception as e:
                    
                    pass
                
                
                after_answer = content.split("</answer>")[-1].strip()
                if after_answer and after_answer != "\n":
                    if len(after_answer) > 1:
                        reward -= (len(after_answer)-1) * 0.001
                    else:
                        reward -= len(after_answer) * 0.001
                    
            rewards.append(reward)
        
        return rewards

    
    
    
    
    
    
    
    
    
    
    
    @staticmethod
    def _validate_scene_classification(json_obj):
        
        required_keys = {"conversation_stage", "scene", "reason", "confidence"}
        return (
            all(key in json_obj for key in required_keys)
            and isinstance(json_obj.get("conversation_stage"), str)
            and (isinstance(json_obj["scene"], str) or json_obj["scene"] is None)
            and isinstance(json_obj["reason"], str)
            and isinstance(json_obj["confidence"], (int, float))
            and 0 <= json_obj["confidence"] <= 1
        )

    @staticmethod
    def _validate_fraud_classification(json_obj):
        
        required_keys = {"conversation_stage", "is_fraud", "reason", "confidence"}
        return (
            all(key in json_obj for key in required_keys)
            and isinstance(json_obj.get("conversation_stage"), str)
            and (isinstance(json_obj["is_fraud"], bool) or json_obj["is_fraud"] is None)
            and isinstance(json_obj["reason"], str)
            and isinstance(json_obj["confidence"], (int, float))
            and 0 <= json_obj["confidence"] <= 1  
        )

    @staticmethod
    def _validate_fraud_type_classification(json_obj):
        
        required_keys = {"conversation_stage", "fraud_type", "reason", "confidence"}
        return (
            all(key in json_obj for key in required_keys)
            and isinstance(json_obj.get("conversation_stage"), str)
            and (isinstance(json_obj["fraud_type"], str) or json_obj["fraud_type"] is None)
            and isinstance(json_obj["reason"], str)
            and isinstance(json_obj["confidence"], (int, float))
            and 0 <= json_obj["confidence"] <= 1  
        )

class NewSceneClassificationORM(ORM):
    def __call__(self, completions, answers, conversation_stage, **kwargs) -> list[float]:
        import os
        import re
        import json
        import datetime
        base_reward = 5.0
        base_penalty = -3.0
        
        
        ALLOWED_SCENES = ["订餐服务", "咨询客服", "预约服务", "交通咨询", "日常购物", "打车服务", "外卖服务", None]
        
        rewards = []
        
        
        current_logs = []
        call_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        audio_indices = kwargs.get("audio_idx", [1]*len(completions))
        audio_lengths = kwargs.get("audio_length", [1]*len(completions))
        
        for content, ans, sta, audio_idx, audio_len in zip(completions, answers, conversation_stage, audio_indices, audio_lengths):
            reward = 0.0
            log_entry = {
                "timestamp": call_timestamp,
                "answer": ans,
                "response": content,
                "error": None,
                "reward": 0.0,
                "details": {
                    "audio_idx": audio_idx,
                    "audio_length": audio_len
                }
            }
            
            try:
                
                answer_text = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
                if not answer_text:
                    rewards.append(0.0)
                    log_entry["error"] = "No answer tag found"
                    current_logs.append(log_entry)
                    continue
                
                
                answer_content = answer_text.group(1).strip()
                
                
                code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", answer_content, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    
                    json_match = re.search(r"(\{.*\})", answer_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        
                        rewards.append(0.0)
                        log_entry["error"] = "No JSON object found in answer"
                        current_logs.append(log_entry)
                        continue
                
                try:
                    json_obj = json.loads(json_str)
                    log_entry["details"]["parsed_json"] = json_obj
                except json.JSONDecodeError:
                    
                    cleaned_json = re.sub(r'[\n\r\t]', ' ', json_str)
                    try:
                        json_obj = json.loads(cleaned_json)
                        log_entry["details"]["parsed_json"] = json_obj
                        log_entry["details"]["json_required_cleaning"] = True
                    except:
                        rewards.append(0.0)
                        log_entry["error"] = "JSON parsing failed after cleaning"
                        current_logs.append(log_entry)
                        continue
                
                scene = json_obj.get("scene")
                reason = json_obj.get("reason")
                
                log_entry["details"]["scene"] = scene
                log_entry["details"]["reason"] = reason
                log_entry["details"]["response_type"] = "json"
                
                
                if scene not in ALLOWED_SCENES and scene is not None:
                    reward = base_penalty  
                    log_entry["details"]["classification_valid"] = False
                    log_entry["details"]["punishment"] = reward
                    log_entry["details"]["punishment_reason"] = f"Scene '{scene}' not in allowed list: {ALLOWED_SCENES}"
                    rewards.append(reward)
                    log_entry["reward"] = reward
                    current_logs.append(log_entry)
                    continue
                
                log_entry["details"]["classification_valid"] = True
                
                
                if scene == ans:
                    reward = base_reward
                    log_entry["details"]["classification_correct"] = True
                elif scene is None and sta == "early_stage":
                    reward = base_reward
                else:
                    reward = 0
                    log_entry["details"]["classification_correct"] = False
                    
            except Exception as e:
                
                log_entry["error"] = f"Error processing response: {str(e)}"
                
            rewards.append(reward)
            log_entry["reward"] = reward
            current_logs.append(log_entry)
        
        
        try:
            
            log_dir = os.path.join(".", "logs", "new_scene_classification_reward_func")
            os.makedirs(log_dir, exist_ok=True)
            
            
            hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
            log_file = os.path.join(log_dir, f"log_{hour_timestamp}.json")
            
            
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, UnicodeError):
                    
                    existing_logs = []
            
            
            existing_logs.extend(current_logs)
            
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing log file: {str(e)}")
        
        return rewards


class NewFraudClassificationORM(ORM):
    def __call__(self, completions, answers, conversation_stage, **kwargs) -> list[float]:
        import os
        import re
        import json
        import datetime
        base_reward = 5.0
        base_penalty = -3.0
        
        rewards = []
        
        
        current_logs = []
        call_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        audio_indices = kwargs.get("audio_idx", [1]*len(completions))
        audio_lengths = kwargs.get("audio_length", [1]*len(completions))
        
        for content, ans, sta, audio_idx, audio_len in zip(completions, answers, conversation_stage, audio_indices, audio_lengths):
            reward = 0.0
            log_entry = {
                "timestamp": call_timestamp,
                "answer": ans,
                "response": content,
                "error": None,
                "reward": 0.0,
                "details": {
                    "audio_idx": audio_idx,
                    "audio_length": audio_len
                }
            }
            
            try:
                
                answer_text = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
                if not answer_text:
                    rewards.append(0.0)
                    log_entry["error"] = "No answer tag found"
                    current_logs.append(log_entry)
                    continue
                
                
                answer_content = answer_text.group(1).strip()
                
                
                code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", answer_content, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    
                    json_match = re.search(r"(\{.*\})", answer_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        
                        rewards.append(0.0)
                        log_entry["error"] = "No JSON object found in answer"
                        current_logs.append(log_entry)
                        continue
                
                try:
                    json_obj = json.loads(json_str)
                    log_entry["details"]["parsed_json"] = json_obj
                except json.JSONDecodeError:
                    
                    cleaned_json = re.sub(r'[\n\r\t]', ' ', json_str)
                    try:
                        json_obj = json.loads(cleaned_json)
                        log_entry["details"]["parsed_json"] = json_obj
                        log_entry["details"]["json_required_cleaning"] = True
                    except:
                        rewards.append(0.0)
                        log_entry["error"] = "JSON parsing failed after cleaning"
                        current_logs.append(log_entry)
                        continue
                
                is_fraud = json_obj.get("is_fraud")
                
                log_entry["details"]["is_fraud"] = is_fraud
                log_entry["details"]["response_type"] = "json"
                
                
                if not isinstance(is_fraud, bool) and is_fraud is not None:
                    reward = base_penalty  
                    log_entry["details"]["is_fraud_valid"] = False
                    log_entry["details"]["punishment"] = reward
                    log_entry["details"]["punishment_reason"] = f"is_fraud must be a boolean value (true/false), got {type(is_fraud).__name__}: {is_fraud}"
                    rewards.append(reward)
                    log_entry["reward"] = reward
                    current_logs.append(log_entry)
                    continue
                
                log_entry["details"]["is_fraud_valid"] = True
                
                
                if (is_fraud is True and ans == "fraud") or (is_fraud is False and ans == "normal"):
                    reward = base_reward
                    log_entry["details"]["classification_correct"] = True
                elif is_fraud is None and sta == "early_stage":
                    reward = base_reward
                else:
                    reward = 0  
                    log_entry["details"]["classification_correct"] = False
                    
            except Exception as e:
                
                log_entry["error"] = f"Error processing response: {str(e)}"
                
            rewards.append(reward)
            log_entry["reward"] = reward
            current_logs.append(log_entry)
        
        
        try:
            
            log_dir = os.path.join(".", "logs", "new_fraud_classification_reward_func")
            os.makedirs(log_dir, exist_ok=True)
            
            
            hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
            log_file = os.path.join(log_dir, f"log_{hour_timestamp}.json")
            
            
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, UnicodeError):
                    
                    existing_logs = []
            
            
            existing_logs.extend(current_logs)
            
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing log file: {str(e)}")
        
        return rewards


class NewFraudTypeClassificationORM(ORM):
    def __call__(self, completions, answers, conversation_stage, **kwargs) -> list[float]:
        import os
        import re
        import json
        import datetime
        base_reward = 5.0
        base_penalty = -3.0
        
        
        ALLOWED_FRAUD_TYPES = ["投资诈骗", "钓鱼诈骗", "身份盗窃", "彩票诈骗", "银行诈骗", "绑架诈骗", "客服诈骗", "邮件诈骗"]
        
        rewards = []
        
        
        current_logs = []
        call_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        audio_indices = kwargs.get("audio_idx", [1]*len(completions))
        audio_lengths = kwargs.get("audio_length", [1]*len(completions))
        
        for content, ans, sta, audio_idx, audio_len in zip(completions, answers, conversation_stage, audio_indices, audio_lengths):
            reward = 0.0
            log_entry = {
                "timestamp": call_timestamp,
                "answer": ans,
                "response": content,
                "error": None,
                "reward": 0.0,
                "details": {
                    "audio_idx": audio_idx,
                    "audio_length": audio_len
                }
            }
            
            try:
                
                answer_text = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
                if not answer_text:
                    rewards.append(0.0)
                    log_entry["error"] = "No answer tag found"
                    current_logs.append(log_entry)
                    continue
                
                
                answer_content = answer_text.group(1).strip()
                
                
                code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", answer_content, re.DOTALL)
                if code_block_match:
                    json_str = code_block_match.group(1)
                else:
                    
                    json_match = re.search(r"(\{.*\})", answer_content, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(1)
                    else:
                        
                        rewards.append(0.0)
                        log_entry["error"] = "No JSON object found in answer"
                        current_logs.append(log_entry)
                        continue
                
                try:
                    json_obj = json.loads(json_str)
                    log_entry["details"]["parsed_json"] = json_obj
                except json.JSONDecodeError:
                    
                    cleaned_json = re.sub(r'[\n\r\t]', ' ', json_str)
                    try:
                        json_obj = json.loads(cleaned_json)
                        log_entry["details"]["parsed_json"] = json_obj
                        log_entry["details"]["json_required_cleaning"] = True
                    except:
                        rewards.append(0.0)
                        log_entry["error"] = "JSON parsing failed after cleaning"
                        current_logs.append(log_entry)
                        continue
                
                fraud_type = json_obj.get("fraud_type")
                reason = json_obj.get("reason")
                
                log_entry["details"]["fraud_type"] = fraud_type
                log_entry["details"]["reason"] = reason
                log_entry["details"]["response_type"] = "json"
                
                
                if fraud_type not in ALLOWED_FRAUD_TYPES and fraud_type is not None:
                    reward = base_penalty  
                    log_entry["details"]["classification_valid"] = False
                    log_entry["details"]["punishment"] = reward
                    log_entry["details"]["punishment_reason"] = f"Fraud type '{fraud_type}' not in allowed list: {ALLOWED_FRAUD_TYPES}"
                    rewards.append(reward)
                    log_entry["reward"] = reward
                    current_logs.append(log_entry)
                    continue
                
                log_entry["details"]["classification_valid"] = True
                
                
                if fraud_type == ans:
                    reward = base_reward
                    log_entry["details"]["classification_correct"] = True
                elif fraud_type is None and sta == "early_stage":
                    reward = base_reward
                else:
                    reward = 0
                    log_entry["details"]["classification_correct"] = False
                    
            except Exception as e:
                
                log_entry["error"] = f"Error processing response: {str(e)}"
                
            rewards.append(reward)
            log_entry["reward"] = reward
            current_logs.append(log_entry)
        
        
        try:
            
            log_dir = os.path.join(".", "logs", "new_fraud_type_classification_reward_func")
            os.makedirs(log_dir, exist_ok=True)
            
            
            hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
            log_file = os.path.join(log_dir, f"log_{hour_timestamp}.json")
            
            
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, UnicodeError):
                    
                    existing_logs = []
            
            
            existing_logs.extend(current_logs)
            
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing log file: {str(e)}")
        
        return rewards


class NewAdaptiveClassificationORM(ORM):
    
    FIELD_TYPES = ["订餐服务", "咨询客服", "预约服务", "交通咨询", "日常购物", "打车服务", "外卖服务"]
    FRAUD_TYPES = ["投资诈骗", "钓鱼诈骗", "身份盗窃", "彩票诈骗", "银行诈骗", "绑架诈骗", "客服诈骗", "邮件诈骗"]
    
    def __call__(self, completions, prompts, answers, conversation_stage, **kwargs) -> list[float]:
        
        scene_orm = NewSceneClassificationORM()
        fraud_orm = NewFraudClassificationORM()
        fraud_type_orm = NewFraudTypeClassificationORM()
        
        
        grouped_indices = {
            "scene": [],  
            "fraud": [],  
            "fraud_type": []  
        }
        
        
        for i, answer in enumerate(answers):
            if answer in self.FIELD_TYPES:
                grouped_indices["scene"].append(i)
            elif answer in ["fraud", "normal"]:
                grouped_indices["fraud"].append(i)
            elif answer in self.FRAUD_TYPES:
                grouped_indices["fraud_type"].append(i)
            else:
                print(f"warning: {answer}")
                
                grouped_indices["scene"].append(i)
        
        
        audio_indices = kwargs.get("audio_idx", [1]*len(prompts))
        audio_lengths = kwargs.get("audio_length", [1]*len(prompts))
        
        
        final_rewards = [0.0] * len(prompts)
        
        
        if grouped_indices["scene"]:
            scene_completions = [completions[i] for i in grouped_indices["scene"]]
            scene_answers = [answers[i] for i in grouped_indices["scene"]]
            scene_kwargs = {
                "audio_idx": [audio_indices[i] for i in grouped_indices["scene"]],
                "audio_length": [audio_lengths[i] for i in grouped_indices["scene"]]
            }
            scene_rewards = scene_orm(scene_completions, scene_answers, conversation_stage, **scene_kwargs)
            for idx, reward in zip(grouped_indices["scene"], scene_rewards):
                final_rewards[idx] = reward
        
        
        if grouped_indices["fraud"]:
            fraud_completions = [completions[i] for i in grouped_indices["fraud"]]
            fraud_answers = [answers[i] for i in grouped_indices["fraud"]]
            fraud_kwargs = {
                "audio_idx": [audio_indices[i] for i in grouped_indices["fraud"]],
                "audio_length": [audio_lengths[i] for i in grouped_indices["fraud"]]
            }
            fraud_rewards = fraud_orm(fraud_completions, fraud_answers, conversation_stage, **fraud_kwargs)
            for idx, reward in zip(grouped_indices["fraud"], fraud_rewards):
                final_rewards[idx] = reward
        
        
        if grouped_indices["fraud_type"]:
            fraud_type_completions = [completions[i] for i in grouped_indices["fraud_type"]]
            fraud_type_answers = [answers[i] for i in grouped_indices["fraud_type"]]
            fraud_type_kwargs = {
                "audio_idx": [audio_indices[i] for i in grouped_indices["fraud_type"]],
                "audio_length": [audio_lengths[i] for i in grouped_indices["fraud_type"]]
            }
            fraud_type_rewards = fraud_type_orm(fraud_type_completions, fraud_type_answers, conversation_stage, **fraud_type_kwargs)
            for idx, reward in zip(grouped_indices["fraud_type"], fraud_type_rewards):
                final_rewards[idx] = reward
        
        return final_rewards
    
class StageClassificationORM(ORM):
    def __call__(self, completions, conversation_stage, **kwargs) -> list[float]:
        import os
        import re
        import json
        import datetime
        
        base_reward = 5.0
        base_penalty = -3.0
        
        
        ALLOWED_STAGES = ["early_stage", "late_stage", "complete"]
        
        rewards = []
        current_logs = []
        call_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        
        audio_indices = kwargs.get("audio_idx", [1]*len(completions))
        audio_lengths = kwargs.get("audio_length", [1]*len(completions))
        
        for content, sta, audio_idx, audio_len in zip(completions, conversation_stage, audio_indices, audio_lengths):
            reward = 0.0
            log_entry = {
                "timestamp": call_timestamp,
                "conversation_stage": sta,
                "response": content,
                "error": None,
                "reward": 0.0,
                "details": {
                    "audio_idx": audio_idx,
                    "audio_length": audio_len,
                }
            }
            
            try:
                
                answer_text = re.search(r"<answer>\s*(.*?)\s*</answer>", content, re.DOTALL)
                if not answer_text:
                    rewards.append(0.0)
                    log_entry["error"] = "No answer tag found"
                    current_logs.append(log_entry)
                    continue
                
                answer_content = answer_text.group(1).strip()
                
                
                code_block_match = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", answer_content, re.DOTALL)
                json_str = code_block_match.group(1) if code_block_match else re.search(r"(\{.*\})", answer_content, re.DOTALL)
                if not json_str:
                    rewards.append(0.0)
                    log_entry["error"] = "No JSON object found in answer"
                    current_logs.append(log_entry)
                    continue
                
                try:
                    json_obj = json.loads(json_str.group(1) if isinstance(json_str, re.Match) else json_str)
                    log_entry["details"]["parsed_json"] = json_obj
                except json.JSONDecodeError:
                    cleaned_json = re.sub(r'[\n\r\t]', ' ', json_str.group(1) if isinstance(json_str, re.Match) else json_str)
                    try:
                        json_obj = json.loads(cleaned_json)
                        log_entry["details"]["parsed_json"] = json_obj
                        log_entry["details"]["json_required_cleaning"] = True
                    except:
                        rewards.append(0.0)
                        log_entry["error"] = "JSON parsing failed after cleaning"
                        current_logs.append(log_entry)
                        continue
                
                
                pre_sta = json_obj.get("conversation_stage")
                scene = json_obj.get("scene")
                is_fraud = json_obj.get("is_fraud")
                fraud_type = json_obj.get("fraud_type")
                reason = json_obj.get("reason")
                confidence = json_obj.get("confidence")
                
                log_entry["details"].update({
                    "conversation_stage": pre_sta,
                    "scene": scene,
                    "is_fraud": is_fraud,
                    "fraud_type": fraud_type,
                    "reason": reason,
                    "confidence": confidence,
                })
                
                
                if pre_sta not in ALLOWED_STAGES and pre_sta is not None:
                    reward = base_penalty
                    log_entry["details"]["stage_valid"] = False
                    log_entry["details"]["punishment"] = reward
                    log_entry["details"]["punishment_reason"] = f"Invalid conversation stage: {pre_sta}"
                    rewards.append(reward)
                    log_entry["reward"] = reward
                    current_logs.append(log_entry)
                    continue
                
                log_entry["details"]["stage_valid"] = True
                if pre_sta == sta:
                    reward = base_reward
                    log_entry["details"]["classification_correct"] = True
                else:
                    reward = 0
                    log_entry["details"]["classification_correct"] = False
                
                
                if (sta == "complete" or pre_sta == "complete") and scene is None and is_fraud is None and fraud_type is None:
                    reward = base_penalty
                    log_entry["details"]["complete_stage_penalty"] = reward
                    log_entry["details"]["penalty_reason"] = (
                        f"Fields cannot be None when conversation_stage is complete"
                    )
                    rewards.append(reward)
                    log_entry["reward"] = reward
                    current_logs.append(log_entry)
                    continue
                    
            except Exception as e:
                log_entry["error"] = f"Error processing response: {str(e)}"
                
            rewards.append(reward)
            log_entry["reward"] = reward
            current_logs.append(log_entry)
        
        
        try:
            log_dir = os.path.join(".", "logs", "multi_task_classification_reward_func")
            os.makedirs(log_dir, exist_ok=True)
            
            hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
            log_file = os.path.join(log_dir, f"log_{hour_timestamp}.json")
            
            existing_logs = []
            if os.path.exists(log_file):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        existing_logs = json.load(f)
                except (json.JSONDecodeError, UnicodeError):
                    existing_logs = []
            
            existing_logs.extend(current_logs)
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(existing_logs, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"Error writing log file: {str(e)}")
        
        return rewards


class ThinkLengthRewardORM(ORM):
    def __init__(self, max_reward_token=200, max_reward=5.0):
        self.max_reward_token = max_reward_token
        self.max_reward = max_reward
    
    def __call__(self, completions, **kwargs) -> list[float]:
        import os
        import re
        import json
        import math
        import datetime
        
        
        rewards = []
        current_logs = []
        call_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for content in completions:
            reward = 0.0
            tokens = 0  
            log_entry = {
                "timestamp": call_timestamp,
                "response": content,
                "error": None,
                "reward": 0.0,
                "details": {}
            }
            
            try:
                
                think_match = re.search(r"<think>\s*([\s\S]*?)\s*</think>", content, re.DOTALL)
                
                if think_match:
                    think_content = think_match.group(1)
                    log_entry["details"]["think_content"] = think_content[:500] + "..." if len(think_content) > 500 else think_content
                    
                    
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', think_content))
                    other_chars = len(think_content) - chinese_chars
                    english_words = len(re.findall(r'[a-zA-Z0-9]+', think_content))
                    punctuation = len(re.findall(r'[,.!?;:()"\'\-，。！？；：（）""'']', think_content))
                    remaining_chars = other_chars - english_words - punctuation
                    tokens = chinese_chars + english_words + punctuation + remaining_chars / 4
                    
                    
                    log_entry["details"]["tokens_estimate"] = tokens
                    log_entry["details"]["chinese_chars"] = chinese_chars
                    log_entry["details"]["english_words"] = english_words
                    log_entry["details"]["punctuation"] = punctuation
                    log_entry["details"]["remaining_chars"] = remaining_chars
                    log_entry["details"]["max_reward_token"] = self.max_reward_token
                    
                    
                    if tokens <= self.max_reward_token:
                        
                        normalized_value = math.log(tokens + 1) / math.log(self.max_reward_token + 1)
                        reward = normalized_value * self.max_reward
                        
                        
                        if tokens > 0:
                            min_reward = 0.1 * self.max_reward
                            if reward < min_reward:
                                reward = min_reward
                                log_entry["details"]["min_reward_applied"] = True
                        
                        log_entry["details"]["normalized_value"] = normalized_value
                    else:
                        
                        reward = self.max_reward
                        log_entry["details"]["max_reward_applied"] = True
                else:
                    log_entry["error"] = "No think section found"
                    
                
                log_entry["reward"] = reward
                
                
                if tokens > 0:
                    log_entry["details"]["token_reward_ratio"] = reward / tokens
                else:
                    log_entry["details"]["token_reward_ratio"] = 0
                
            except Exception as e:
                log_entry["error"] = f"Error processing response: {str(e)}"
            
            rewards.append(reward)
            current_logs.append(log_entry)
        
        
        try:
            
            import torch.distributed as dist
            if dist.is_initialized():
                
                rank = dist.get_rank()
                is_main_process = (rank == 0)
            else:
                
                is_main_process = True
        except ImportError:
            
            is_main_process = True
        
        
        if is_main_process:
            try:
                
                log_dir = os.path.join(".", "logs", "think_length_reward_func")
                os.makedirs(log_dir, exist_ok=True)
                
                
                hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
                log_file = os.path.join(log_dir, f"log_{hour_timestamp}.json")
                
                
                existing_logs = []
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            existing_logs = json.load(f)
                    except (json.JSONDecodeError, UnicodeError):
                        
                        existing_logs = []
                
                
                existing_logs.extend(current_logs)
                
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_logs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error writing log file: {str(e)}")
        
        return rewards


class ThinkLengthPenaltyORM(ORM):
    def __init__(self, penalty_start_token=300, max_penalty=5.0):
        self.penalty_start_token = penalty_start_token
        self.max_penalty = max_penalty
    
    def __call__(self, completions, **kwargs) -> list[float]:
        import os
        import re
        import json
        import math
        import datetime
        
        
        rewards = []
        current_logs = []
        call_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        for content in completions:
            reward = 0.0
            log_entry = {
                "timestamp": call_timestamp,
                "response": content,
                "error": None,
                "reward": 0.0,
                "details": {}
            }
            
            try:
                
                think_match = re.search(r"<think>\s*([\s\S]*?)\s*</think>", content, re.DOTALL)
                
                if think_match:
                    think_content = think_match.group(1)
                    log_entry["details"]["think_content"] = think_content[:500] + "..." if len(think_content) > 500 else think_content
                    
                    
                    chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', think_content))
                    other_chars = len(think_content) - chinese_chars
                    english_words = len(re.findall(r'[a-zA-Z0-9]+', think_content))
                    punctuation = len(re.findall(r'[,.!?;:()"\'\-，。！？；：（）""'']', think_content))
                    remaining_chars = other_chars - english_words - punctuation
                    tokens = chinese_chars + english_words + punctuation + remaining_chars / 4
                    
                    
                    log_entry["details"]["tokens_estimate"] = tokens
                    log_entry["details"]["chinese_chars"] = chinese_chars
                    log_entry["details"]["english_words"] = english_words
                    log_entry["details"]["punctuation"] = punctuation
                    log_entry["details"]["remaining_chars"] = remaining_chars
                    log_entry["details"]["penalty_start_token"] = self.penalty_start_token
                    
                    
                    if tokens <= self.penalty_start_token:
                        
                        reward = 0.0
                        log_entry["details"]["no_penalty_applied"] = True
                    else:
                        
                        excess_tokens = tokens - self.penalty_start_token
                        
                        
                        
                        base_value = 1000  
                        normalized_value = math.log(excess_tokens + 10) / math.log(base_value)
                        penalty = normalized_value * self.max_penalty
                        
                        
                        min_penalty = 0.1 * self.max_penalty
                        if penalty < min_penalty:
                            penalty = min_penalty
                            log_entry["details"]["min_penalty_applied"] = True
                            
                        
                        if penalty > self.max_penalty:
                            penalty = self.max_penalty
                            log_entry["details"]["max_penalty_applied"] = True
                            
                        reward = -penalty  
                        log_entry["details"]["normalized_value"] = normalized_value
                        log_entry["details"]["excess_tokens"] = excess_tokens
                else:
                    log_entry["error"] = "No think section found"
                    
                log_entry["reward"] = reward
                if "tokens_estimate" in log_entry["details"] and tokens > self.penalty_start_token:
                    log_entry["details"]["token_penalty_ratio"] = abs(reward) / (tokens - self.penalty_start_token) if tokens > self.penalty_start_token else 0
                
            except Exception as e:
                log_entry["error"] = f"Error processing response: {str(e)}"
            
            rewards.append(reward)
            current_logs.append(log_entry)
        
        
        try:
            
            import torch.distributed as dist
            if dist.is_initialized():
                
                rank = dist.get_rank()
                is_main_process = (rank == 0)
            else:
                
                is_main_process = True
        except ImportError:
            
            is_main_process = True
        
        
        if is_main_process:
            try:
                
                log_dir = os.path.join(".", "logs", "think_length_penalty_func")
                os.makedirs(log_dir, exist_ok=True)
                
                
                hour_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H")
                log_file = os.path.join(log_dir, f"log_{hour_timestamp}.json")
                
                
                existing_logs = []
                if os.path.exists(log_file):
                    try:
                        with open(log_file, 'r', encoding='utf-8') as f:
                            existing_logs = json.load(f)
                    except (json.JSONDecodeError, UnicodeError):
                        
                        existing_logs = []
                
                
                existing_logs.extend(current_logs)
                
                
                with open(log_file, 'w', encoding='utf-8') as f:
                    json.dump(existing_logs, f, ensure_ascii=False, indent=2)
            except Exception as e:
                logger.error(f"Error writing log file: {str(e)}")
        
        return rewards



orms['prompt_format'] = PromptFormatORM
orms['new_scene_classification'] = NewSceneClassificationORM
orms['new_fraud_classification'] = NewFraudClassificationORM
orms['new_fraud_type_classification'] = NewFraudTypeClassificationORM
orms['new_adaptive_classification'] = NewAdaptiveClassificationORM
orms['stage_classification'] = StageClassificationORM
orms['think_length_reward_200'] = lambda: ThinkLengthRewardORM(max_reward_token=200, max_reward=5.0)
orms['think_length_reward_300'] = lambda: ThinkLengthRewardORM(max_reward_token=300, max_reward=5.0)
orms['think_length_penalty_300'] = lambda: ThinkLengthPenaltyORM(penalty_start_token=300, max_penalty=5.0)
orms['think_length_penalty_500'] = lambda: ThinkLengthPenaltyORM(penalty_start_token=500, max_penalty=7.0)