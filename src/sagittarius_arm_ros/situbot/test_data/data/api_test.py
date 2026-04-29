import os
import json
from openai import OpenAI

# ===================== 1. 配置基础参数 =====================
client = OpenAI(
    api_key=os.environ.get("DASHSCOPE_API_KEY",""),
    base_url="https://api.apiyi.com/v1"
)

INPUT_FILE = "cleaned_data.json"
OUTPUT_FILE = "arrangement_result.json"

# 坐标约束
COORD_CONSTRAINTS = {
    "x": {"min": 0.15, "max": 0.75},  # x：深度（-0.5=最靠近人，0.5=最远离人）
    "y": {"min": -0.4, "max": 0.4},       # y：宽度（0=人的右侧，1=人的左侧）
    "z": 0.1
}

# ===================== 2. 结构化精简版一体化Prompt =====================
SYSTEM_PROMPT = f"""任务：桌面物品空间布局规划
输出要求：仅返回JSON对象，无任何额外文字

一、坐标约束
1. x有效范围：0.15 ≤ x ≤ 0.75
2. y有效范围：-0.4 ≤ y ≤ 0.4，**y的绝对值严禁大于0.4**
3. 禁行区域：y ∈ [-0.1, 0.1]，禁止放置任何物品
4. z固定为：{COORD_CONSTRAINTS['z']}

二、物品角色（仅3类）
1. prominent（核心物品）
   有效区域：x ∈ [0.3, 0.6]，y ∈ [-0.4, -0.15] 或 [0.15, 0.4]
2. accessible（辅助物品）
   有效区域：x ∈ [0.2, 0.7]，y ∈ [-0.4, -0.15] 或 [0.15, 0.4]
3. peripheral（次要物品）
   有效区域：x ∈ [0.15, 0.3] 或 [0.6, 0.75]，y ∈ [-0.4, -0.15] 或 [0.15, 0.4]

三、布局规则
1. 任意两个物品间距≥0.1米，禁止坐标重叠
2. 语义相关物品可就近摆放，无需紧密聚集

四、输出JSON格式
{{
  "scenario_id": "场景ID",
  "situation": "场景描述",
  "object_arrangements": [
    {{
      "name": "物品名称",
      "role": "prominent/accessible/peripheral",
      "coordinate": {{"x": 数值, "y": 数值, "z": {COORD_CONSTRAINTS['z']}}},
      "reasoning": "布局理由"
    }}
  ],
  "layout_summary": "整体布局说明"
}}
"""

# ===================== 3. 读取数据 =====================
def load_cleaned_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# ===================== 4. 构建用户Prompt（强化场景信息传递） =====================
def build_user_prompt(cleaned_data_item):
    scenario_id = cleaned_data_item["scenario_id"]
    situation = cleaned_data_item["situation"]
    object_coordinates = cleaned_data_item["object_coordinates"]

    # 补充物品初始状态说明（区分可移动/固定）
    obj_info = []
    for obj in object_coordinates:
        graspable = "可移动" if obj.get("graspable", True) else "固定位置（不可移动）"
        obj_info.append(
            f"- {obj['name']}：{graspable}，初始坐标（x={obj['coordinate']['x']}, y={obj['coordinate']['y']}）"
        )
    obj_info_str = "\n".join(obj_info)

    user_prompt = f"""请基于以下场景信息完成物品摆放规划，严格遵循系统提示的所有规则：
场景ID：{scenario_id}
场景描述：{situation}
候选物品及初始状态：
{obj_info_str}

强制要求：
1. 坐标必须符合 x∈[{COORD_CONSTRAINTS['x']['min']}, {COORD_CONSTRAINTS['x']['max']}]、y∈[{COORD_CONSTRAINTS['y']['min']}, {COORD_CONSTRAINTS['y']['max']}]
2. 固定物品坐标保持初始值，仅更新角色和理由
3. 输出仅保留JSON，无任何额外内容"""
    return user_prompt

# ===================== 5. 主函数 =====================
def main():
    cleaned_data = load_cleaned_data(INPUT_FILE)
    if not cleaned_data:
        print("❌ 输入文件为空")
        return

    results = []
    for item in cleaned_data:
        print(f"\n🔍 处理场景：{item['scenario_id']}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(item)}
        ]

        try:
            print("📡 调用LLM API...")
            completion = client.chat.completions.create(
                model="qwen-long",
                stream=False,
                messages=messages,
                temperature=0.7,
                response_format={"type": "json_object"}  # 强制JSON输出（GLM支持该参数）
            )

            response_content = completion.choices[0].message.content.strip()
            print(f"✅ LLM返回结果：\n{response_content}")

            # 解析JSON（增加容错：去除可能的首尾空白/引号）
            result = json.loads(response_content)
            results.append(result)

        except json.JSONDecodeError as e:
            print(f"❌ JSON解析失败：{str(e)}，返回内容：{response_content}")
            continue
        except Exception as e:
            print(f"❌ 处理失败：{str(e)}")
            continue

    # 保存结果
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"\n📁 结果已保存到：{OUTPUT_FILE}，共处理 {len(results)} 个场景")

if __name__ == "__main__":
    main()