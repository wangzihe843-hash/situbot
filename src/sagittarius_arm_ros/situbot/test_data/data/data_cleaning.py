import yaml
import json

# ===================== 配置项 =====================
INPUT_FILE = "input.txt"  # 你的原始数据文件
OUTPUT_FILE = "cleaned_data.json"  # 清洗后输出文件
# 临时填充场景信息（后续你直接替换即可）
TEMP_SCENE = {
    "scenario_id": "TEST01",
    "situation": "tabletop object arrangement test",
    "level": "functional"
}


# ==================================================

def clean_ros_data():
    # 读取原始数据
    try:
        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        print(f"错误：未找到输入文件 {INPUT_FILE}")
        return

    # 按 --- 分割数据块
    blocks = [block.strip() for block in content.split("---") if block.strip()]
    if not blocks:
        print("未检测到有效数据")
        return

    # 收集所有物品（去重）
    object_names = []  # 纯英文物品名列表
    object_coords = []  # 带坐标的详情

    for block in blocks:
        try:
            data = yaml.safe_load(block)
            objects = data.get("objects", [])

            if not objects:
                continue  # 跳过无物品的帧

            for obj in objects:
                obj_name = obj.get("name", "")
                # 提取坐标（保留3位小数）
                coords = {
                    "x": round(obj.get("x", 0.0), 3),
                    "y": round(obj.get("y", 0.0), 3),
                    "z": round(obj.get("z", 0.0), 3)
                }
                # 去重
                if obj_name and obj_name not in object_names:
                    object_names.append(obj_name)
                    object_coords.append({
                        "name": obj_name,
                        "coordinate": coords
                    })
        except Exception as e:
            print(f"解析单条数据失败: {e}")
            continue

    if not object_names:
        print("未提取到有效物品")
        return

    # 构建最终JSON结构（完全匹配你的格式）
    final_result = [
        {
            "scenario_id": TEMP_SCENE["scenario_id"],
            "situation": TEMP_SCENE["situation"],
            "level": TEMP_SCENE["level"],
            "candidate_objects": object_names,
            # 额外保留坐标（你要求必须加）
            "object_coordinates": object_coords
        }
    ]

    # 保存文件
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_result, f, ensure_ascii=False, indent=4)

    print(f"✅ 清洗完成！文件已保存至: {OUTPUT_FILE}")
    print(f"📦 提取到的物品（英文）: {object_names}")


if __name__ == "__main__":
    clean_ros_data()