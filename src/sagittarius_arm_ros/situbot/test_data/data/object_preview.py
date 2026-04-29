import json
import matplotlib.pyplot as plt

# ===================== 配置项 =====================
INPUT_JSON = "arrangement_result.json"
# 坐标范围
X_MIN, X_MAX = 0.15, 0.75
Y_MIN, Y_MAX = -0.4, 0.4

# 角色样式（仅3类，无remove）
ROLE_STYLE = {
    "prominent": {"color": "#e63946", "marker": "o", "label": "核心物品"},
    "accessible": {"color": "#457b9d", "marker": "s", "label": "辅助物品"},
    "peripheral": {"color": "#8ecae6", "marker": "D", "label": "次要物品"}
}
# ==================================================

def plot_layout():
    # 读取数据
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)[0]

    scenario_id = data["scenario_id"]
    situation = data["situation"]
    objects = data["object_arrangements"]

    # 画布设置
    plt.figure(figsize=(10, 8), dpi=100)
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    # 遍历物品，原样绘制原始坐标
    for obj in objects:
        name = obj["name"]
        role = obj["role"]
        x = obj["coordinate"]["x"]
        y = obj["coordinate"]["y"]

        # 过滤非法坐标
        if not (X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX):
            print(f"⚠️ 跳过非法坐标：{name} (x={x}, y={y})")
            continue

        # 绘制原始点位，无任何偏移
        style = ROLE_STYLE[role]
        plt.scatter(x, y, c=style["color"], marker=style["marker"], s=120, edgecolors="black", linewidth=1)
        plt.annotate(name, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=10)

    # 图表样式
    plt.title(f"桌面布局预览\n{scenario_id} | {situation}", fontsize=14, pad=20)
    plt.xlabel("X 轴 (靠近人 ←  → 远离人)", fontsize=12)
    plt.ylabel("Y 轴 (右侧 ←  → 左侧)", fontsize=12)
    plt.xlim(X_MIN - 0.05, X_MAX + 0.05)
    plt.ylim(Y_MIN - 0.05, Y_MAX + 0.05)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend(loc="upper right", fontsize=11)
    plt.tight_layout()

    # 保存并显示
    plt.savefig("layout_preview.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("✅ 点状图已保存为 layout_preview.png")

if __name__ == "__main__":
    plot_layout()