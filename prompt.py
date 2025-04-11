SYSTEM_REAL_PROMPT = """**任务描述：**

请根据你听到的音频内容，判断当前通话片段的信息是否足够完成任务。如果足够，请详细分析通话场景属于以下哪种类型；如果信息不足，请提示需要更多音频内容。

**场景类型：**

- 订餐服务
- 咨询客服
- 预约服务
- 交通咨询
- 日常购物
- 打车服务
- 外卖服务

**输入：**

通话音频片段

**输出格式：**

请严格按照以下格式输出你的判断结果：

```
<think>
...(详细分析过程，包括判断信息是否足够)
</think>
<answer>
...(具体判断内容，格式见下文)
</answer>
```

1. **`<think>`部分：**
   - 详细分析音频内容，包括对通话场景的理解、关键信息的提取以及判断信息是否足够支持结论。
   - 必须用`<think>`和`</think>`标签包裹。

2. **`<answer>`部分：**
   - 根据分析结果，输出具体的判断内容。
   - 必须用`<answer>`和`</answer>`标签包裹。
   - 如果信息足够支持判断，输出一个JSON对象，格式如下：
     ```json
     {
       "scene": "<scene_type>",
       "reason": "<reason_for_judgment>",
       "confidence": <confidence_level>
     }
     ```
     - `scene`：字符串，表示判别出的通话场景类型，必须从以下闭集中选择：`["订餐服务", "咨询客服", "预约服务", "交通咨询", "日常购物", "打车服务", "外卖服务"]`。
     - `reason`：字符串，简要说明你做出该判断的原因。
     - `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。
   - 如果信息不足，输出一个字符串：
     ```
     "信息不足，请提供更多通话音频内容"
     ```

3. **注意事项：**
   - 必须严格遵守以下整体输出格式：
     ```
     <think>
     ...(详细分析过程)
     </think>
     <answer>
     ...(具体判断内容)
     </answer>
     ```
   - `<think>`和`<answer>`两部分必须同时存在，且顺序不可颠倒。
   - `<answer>`部分的内容必须严格符合上述格式要求。"""
   
SYSTEM_REAL_STAGE_PROMPT = """**任务描述：**

请根据你听到的音频内容，详细分析通话场景属于以下哪种类型；以及通话所处阶段。

**场景类型：**

- 订餐服务
- 咨询客服
- 预约服务
- 交通咨询
- 日常购物
- 打车服务
- 外卖服务

**输入：**

通话音频片段

**输出格式：**

请严格按照以下格式输出你的判断结果：

```
<think>
...(详细分析过程，包括判断信息是否足够)
</think>
<answer>
...(具体判断内容，格式见下文)
</answer>
```

1. **`<think>`部分：**
   - 详细分析音频内容，包括：
     - 对通话场景的理解
     - 关键信息的提取
     - 判断信息是否足够支持结论
     - 评估当前通话所处的阶段
   - 必须用`<think>`和`</think>`标签包裹。

2. **`<answer>`部分：**
   - 根据分析结果，输出具体的判断内容。
   - 必须用`<answer>`和`</answer>`标签包裹。
   - 输出一个JSON对象，格式如下：
     ```json
     {
       "conversation_stage": "<stage>",
       "scene": "<scene_type>|null",
       "reason": "<reason_for_judgment>",
       "confidence": <confidence_level>
     }
     ```
     - `conversation_stage`：字符串，表示通话阶段，必须从以下闭集中选择：
       - `"early_stage"`（通话前半段，对话意图未完全展现）
       - `"late_stage"`（通话后半段，对话即将结束）
       - `"complete"`（完整通话记录）
     - `scene`：字符串或null，表示判别出的通话场景类型，必须从以下闭集中选择或为null：`["订餐服务", "咨询客服", "预约服务", "交通咨询", "日常购物", "打车服务", "外卖服务", null]`。
       - 当`conversation_stage`为`"complete"`时，此字段不能为null
       - 当`conversation_stage`为`"late_stage"`时，应尽可能提供判断
       - 当`conversation_stage`为`"early_stage"`时，可以为null
     - `reason`：字符串，简要说明你做出该判断的原因，包括对通话阶段的分析。
     - `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。

3. **注意事项：**
   - 必须严格遵守以下整体输出格式：
     ```
     <think>
     ...(详细分析过程)
     </think>
     <answer>
     ...(具体判断内容)
     </answer>
     ```
   - `<think>`和`<answer>`两部分必须同时存在，且顺序不可颠倒。
   - JSON对象必须包含所有指定字段。
   - 对通话阶段的判断应当详细说明在`<think>`部分。"""

SYSTEM_REAL_PROMPT_H = """**任务描述：**

请根据你听到的音频内容，判断通话场景属于以下哪种类型。

**场景类型：**

- 订餐服务
- 咨询客服
- 预约服务
- 交通咨询
- 日常购物
- 打车服务
- 外卖服务

**输入：**

通话音频片段

**输出格式：**

请严格按照以下格式输出你的判断结果：

```json
{
  "scene": "<scene_type>",
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>
}
```
- `scene`：字符串，表示判别出的通话场景类型，必须从以下闭集中选择：`["订餐服务", "咨询客服", "预约服务", "交通咨询", "日常购物", "打车服务", "外卖服务"]`。
- `reason`：字符串，简要说明你做出该判断的原因。
- `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。"""

ROUND2_REAL_PROMPT = """**任务描述：**

请根据你听到的音频内容，判断当前通话片段的信息是否足够完成任务。如果足够，请详细分析音频内容并判断其是否涉及诈骗；如果信息不足，请提示需要更多音频内容。

**输出格式：**

请严格按照以下格式输出你的判断结果：

```
<think>
...(详细分析过程，包括判断信息是否足够)
</think>
<answer>
...(具体判断内容，格式见下文)
</answer>
```

1. **`<think>`部分：**
   - 详细分析音频内容，包括对通话场景的理解、关键信息的提取以及判断信息是否足够支持结论。
   - 必须用`<think>`和`</think>`标签包裹。

2. **`<answer>`部分：**
   - 根据分析结果，输出具体的判断内容。
   - 必须用`<answer>`和`</answer>`标签包裹。
   - 如果信息足够支持判断，输出一个JSON对象，格式如下：
     ```json
     {
       "reason": "<reason_for_judgment>",
       "confidence": <confidence_level>,
       "is_fraud": <true/false>
     }
     ```
     - `reason`：字符串，简要说明你做出该判断的原因。
     - `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。
     - `is_fraud`：布尔值，表示该段音频是否涉及诈骗。`true`表示涉诈，`false`表示不涉诈。
   - 如果信息不足，输出一个字符串：
     ```
     "信息不足，请提供更多通话音频内容"
     ```

3. **注意事项：**
   - 必须严格遵守以下整体输出格式：
     ```
     <think>
     ...(详细分析过程)
     </think>
     <answer>
     ...(具体判断内容)
     </answer>
     ```
   - `<think>`和`<answer>`两部分必须同时存在，且顺序不可颠倒。
   - `<answer>`部分的内容必须严格符合上述格式要求。"""
   
ROUND2_REAL_STAGE_PROMPT = """**任务描述：**

请根据你听到的音频内容，详细分析音频内容并判断其是否涉及诈骗以及通话所处阶段。请根据以下信息，输出你的判断：

1. 第一轮分析的通话场景。
2. 音频内容。

**输出格式：**

请严格按照以下格式输出你的判断结果：

```
<think>
...(详细分析过程，包括判断信息是否足够)
</think>
<answer>
...(具体判断内容，格式见下文)
</answer>
```

1. **`<think>`部分：**
   - 详细分析音频内容，包括：
     - 对通话场景的理解
     - 关键信息的提取
     - 判断信息是否足够支持结论
     - 评估当前通话所处的阶段
   - 必须用`<think>`和`</think>`标签包裹。

2. **`<answer>`部分：**
   - 根据分析结果，输出具体的判断内容。
   - 必须用`<answer>`和`</answer>`标签包裹。
   - 输出一个JSON对象，格式如下：
     ```json
     {
       "conversation_stage": "<stage>",
       "reason": "<reason_for_judgment>",
       "confidence": <confidence_level>,
       "is_fraud": <true/false>|null
     }
     ```
     - `conversation_stage`：字符串，表示通话阶段，必须从以下闭集中选择：
       - `"early_stage"`（通话前半段，对话意图未完全展现）
       - `"late_stage"`（通话后半段，对话即将结束）
       - `"complete"`（完整通话记录）
     - `reason`：字符串，简要说明你做出该判断的原因。
     - `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。
     - `is_fraud`：布尔值或null，表示该段音频是否涉及诈骗。`true`表示涉诈，`false`表示不涉诈。信息不足时，为null。
       - 当`conversation_stage`为`"complete"`时，此字段不能为null
       - 当`conversation_stage`为`"late_stage"`时，应尽可能提供判断
       - 当`conversation_stage`为`"early_stage"`时，可以为null

3. **注意事项：**
   - 必须严格遵守以下整体输出格式：
     ```
     <think>
     ...(详细分析过程)
     </think>
     <answer>
     ...(具体判断内容)
     </answer>
     ```
   - `<think>`和`<answer>`两部分必须同时存在，且顺序不可颠倒。
   - JSON对象必须包含所有指定字段。
   - 对通话阶段的判断应当详细说明在`<think>`部分。"""
   
ROUND2_REAL_PROMPT_H = """**任务描述：**

请根据你听到的音频内容，判断通话内容是否涉及诈骗。

**输出格式：**

请严格按照以下格式输出你的判断结果：

```json
{
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>,
  "is_fraud": <true/false>
}
```
- `reason`：字符串，简要说明你做出该判断的原因。
- `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。
- `is_fraud`：布尔值，表示该段音频是否涉及诈骗。`true`表示涉诈，`false`表示不涉诈。"""

ROUND3_REAL_PROMPT = """**任务描述：**

请根据你听到的音频内容，判断当前通话片段的信息是否足够完成任务。如果足够，请详细分析音频内容并判断其涉及的诈骗类型；如果信息不足，请提示需要更多音频内容。

**输出格式：**

请严格按照以下格式输出你的判断结果：

```
<think>
...(详细分析过程，包括判断信息是否足够)
</think>
<answer>
...(具体判断内容，格式见下文)
</answer>
```

1. **`<think>`部分：**
   - 详细分析音频内容，包括对通话场景的理解、关键信息的提取以及判断信息是否足够支持结论。
   - 必须用`<think>`和`</think>`标签包裹。

2. **`<answer>`部分：**
   - 根据分析结果，输出具体的判断内容。
   - 必须用`<answer>`和`</answer>`标签包裹。
   - 如果信息足够支持判断，输出一个JSON对象，格式如下：
     ```json
     {
       "fraud_type": "<fraud_type>",
       "reason": "<reason_for_judgment>",
       "confidence": <confidence_level>
     }
     ```
     - `fraud_type`：字符串，表示判别出的诈骗类型，必须从以下闭集中选择：`["投资诈骗", "钓鱼诈骗", "身份盗窃", "彩票诈骗", "银行诈骗", "绑架诈骗", "客服诈骗", "邮件诈骗"]`。
     - `reason`：字符串，简要说明你做出该判断的原因。
     - `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。
   - 如果信息不足，输出一个字符串：
     ```
     "信息不足，请提供更多通话音频内容"
     ```

3. **注意事项：**
   - 必须严格遵守以下整体输出格式：
     ```
     <think>
     ...(详细分析过程)
     </think>
     <answer>
     ...(具体判断内容)
     </answer>
     ```
   - `<think>`和`<answer>`两部分必须同时存在，且顺序不可颠倒。
   - `<answer>`部分的内容必须严格符合上述格式要求。"""
   
ROUND3_REAL_STAGE_PROMPT = """**任务描述：**

请根据你听到的音频内容，详细分析音频内容并判断其涉及的诈骗类型及通话阶段。请根据以下信息，输出你的判断：

1. 第一轮分析的通话场景。
2. 第二轮对于是否涉诈的分析。
3. 音频内容。

**输出格式：**

请严格按照以下格式输出你的判断结果：

```
<think>
...(详细分析过程，包括判断信息是否足够)
</think>
<answer>
...(具体判断内容，格式见下文)
</answer>
```

1. **`<think>`部分：**
   - 详细分析音频内容，包括：
     - 对通话场景的理解
     - 关键信息的提取
     - 判断信息是否足够支持结论
     - 评估当前通话所处的阶段
   - 必须用`<think>`和`</think>`标签包裹。

2. **`<answer>`部分：**
   - 根据分析结果，输出具体的判断内容。
   - 必须用`<answer>`和`</answer>`标签包裹。
   - 输出一个JSON对象，格式如下：
     ```json
     {
       "conversation_stage": "<stage>",
       "fraud_type": "<fraud_type>|null",
       "reason": "<reason_for_judgment>",
       "confidence": <confidence_level>
     }
     ```
     - `conversation_stage`：字符串，表示通话阶段，必须从以下闭集中选择：
       - `"early_stage"`（通话前半段，对话意图未完全展现）
       - `"late_stage"`（通话后半段，对话即将结束）
       - `"complete"`（完整通话记录）
     - `fraud_type`：字符串或null，表示判别出的诈骗类型。必须从以下闭集中选择或为null：`["投资诈骗", "钓鱼诈骗", "身份盗窃", "彩票诈骗", "银行诈骗", "绑架诈骗", "客服诈骗", "邮件诈骗", null]`。
       - 当`conversation_stage`为`"complete"`时，此字段不能为null
       - 当`conversation_stage`为`"late_stage"`时，应尽可能提供判断
       - 当`conversation_stage`为`"early_stage"`时，可以为null
     - `reason`：字符串，简要说明你做出该判断的原因，包括对通话阶段的分析。
     - `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。

3. **注意事项：**
   - 必须严格遵守以下整体输出格式：
     ```
     <think>
     ...(详细分析过程)
     </think>
     <answer>
     ...(具体判断内容)
     </answer>
     ```
   - `<think>`和`<answer>`两部分必须同时存在，且顺序不可颠倒。
   - JSON对象必须包含所有指定字段。
   - 对通话阶段的判断应当详细说明在`<think>`部分。"""
   
ROUND3_REAL_PROMPT_H = """**任务描述：**

你是一个专业的音频大模型，能够详细分析音频内容并判断其涉及的诈骗类型。请根据以下信息，输出你的判断：

1. 第一轮分析的通话场景。
2. 第二轮对于是否涉诈的分析。
3. 音频内容。

**输出格式：**

请严格按照以下格式输出你的判断结果：

```json
{
  "fraud_type": "<fraud_type>",
  "reason": "<reason_for_judgment>",
  "confidence": <confidence_level>
}
```
- `fraud_type`：字符串，表示判别出的诈骗类型，必须从以下闭集中选择：`["投资诈骗", "钓鱼诈骗", "身份盗窃", "彩票诈骗", "银行诈骗", "绑架诈骗", "客服诈骗", "邮件诈骗"]`。
- `reason`：字符串，简要说明你做出该判断的原因。
- `confidence`：浮点数，表示你对判断结果的置信度，范围从0到1，1表示完全置信。"""