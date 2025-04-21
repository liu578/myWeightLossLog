import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np
from scipy.optimize import root_scalar

# 设置暗色主题
plt.style.use('dark_background')
plt.rcParams['font.sans-serif'] = ['Heiti TC']
# start_date = datetime.now() - timedelta(days=datetime.now().weekday())
start_date = datetime(2025, 3, 24) # 开始日期, 2025年3月24日

# === 用户输入部分 ===
height = 172  # 身高
start_weight = 90  # 当前体重
normal_weight = 73.7 # 正常体重的上限
target_weight = 62.1  # 最终目标体重, 建议根据bmi来计算，这里我取bmi21，我的身高的最佳体重就是62.1
weekly_percent = 0.01  # 每周减去当前体重的1%，一个月/4周的减重应该控制在总体重的5%以内，减的过快会造成皮肤松弛，不可逆！！！
weekly_percent_upper_limit = 0.012 #上限设置在1.2%
# 计算达到目标体重所需的周数
def calculate_weeks_needed(start, target, percent):
    weeks = 0
    current = start
    while current > target:
        current = current * (1 - percent)
        weeks += 1
    return weeks

total_weeks = calculate_weeks_needed(start_weight, target_weight, weekly_percent)

# ✅ 手动填写你的实际体重（每周更新一个）
# 示例：actual_weights = [89.2, 88.3, 87.9]  # 前三周的体重
actual_weights = [
    89.1, # 第一周周末的体重
    87.1, # 第二周周末的体重
    85.2, # 第三周周末的体重
    83.8, # 第四周周末的体重
    # 87.1, # 第五周周末的体重
    # 86.7, # 第六周周末的体重
    # 86.3, # 第七周周末的体重
    # 85.9, # 第八周周末的体重
    # 85.5, # 第九周周末的体重
    # 85.1, # 第十周周末的体重
    # 84.7, # 第十一周周末的体重
    # 84.3, # 第十二周周末的体重
    # 83.9, # 第十三周周末的体重
    # 83.5, # 第十四周周末的体重
    # 83.1, # 第十五周周末的体重
    # 71, # 第十六周周末的体重
    # 70.5, # 第十七周周末的体重
    # 70, # 第十八周周末的体重
    # 69.5, # 第十九周周末的体重
    # 69, # 第二十周周末的体重
    # 68.5, # 第二十一周周末的体重
    # 68, # 第二十二周周末的体重
    # 67.5, # 第二十三周周末的体重
    # 67, # 第二十四周周末的体重
    # 66.5, # 第二十五周周末的体重
    # 66, # 第二十六周周末的体重
    # 65.5, # 第二十七周周末的体重
    # 65, # 第二十八周周末的体重
    # 64.5, # 第二十九周周末的体重
    # 64, # 第三十周周末的体重
]
# === end of user input ===

# 输入验证
if target_weight >= start_weight:
    raise ValueError("目标体重必须小于起始体重")
if height <= 0:
    raise ValueError("身高必须为正数")

total_loss = start_weight - target_weight

# 构造计划体重表
weeks = []
week_index = 0
current_weight = start_weight

for week in range(total_weeks):
    # 计算本周应该减去的重量
    weekly_loss = current_weight * weekly_percent  # 每周减去当前体重的1%
    
    current_weight -= weekly_loss
    week_start = start_date + timedelta(weeks=week_index)
    week_end = week_start + timedelta(days=6)
    weeks.append({
        "第几周": week_index + 1,
        "终止日": week_end.strftime("%m-%d"),
        "计划体重": round(current_weight, 1),
        # "周减重": round(weekly_loss, 2)
    })
    week_index += 1

print(weeks)

df = pd.DataFrame(weeks)

# 补齐长度
while len(actual_weights) < len(df):
    actual_weights.append(None)

df["实际体重"] = actual_weights

# 计算累计达标率
cumulative_success = []
cumulative_total = []
success_count = 0
total_count = 0
weekly_contributions = []

for i, row in df.iterrows():
    y = row["实际体重"]
    if pd.notna(y):
        total_count += 1
        # 获取上周的实际体重
        prev_weight = actual_weights[i-1] if i > 0 else start_weight
        if pd.notna(prev_weight):
            # 计算实际减重比例
            actual_loss_percent = (prev_weight - y) / prev_weight
            # 判断是否达标（是否减重超过1%）
            if actual_loss_percent >= weekly_percent and actual_loss_percent <= weekly_percent_upper_limit:
                success_count += 1
                weekly_contributions.append('o')  # 达标，正贡献
            elif actual_loss_percent > weekly_percent_upper_limit:
                weekly_contributions.append('^')  # 减的太快，负贡献
            else:
                weekly_contributions.append('v')  # 未达标，负贡献
        else:
            # 第一周特殊处理
            if (start_weight - y) / start_weight >= weekly_percent and (start_weight - y) / start_weight <= weekly_percent_upper_limit:
                success_count += 1
                weekly_contributions.append('o')
            else:
                weekly_contributions.append('^')
        cumulative_success.append(success_count)
        cumulative_total.append(total_count)
    else:
        cumulative_success.append(None)
        cumulative_total.append(None)
        weekly_contributions.append(None)

# 计算每周的达成率
cumulative_rates = []
for s, t in zip(cumulative_success, cumulative_total):
    if t is not None and t > 0:  # 添加 t > 0 的检查
        cumulative_rates.append((s/t)*100)
    else:
        cumulative_rates.append(None)

# BMI计算函数
def calculate_bmi(weight, height_cm):
    height_m = height_cm / 100
    return weight / (height_m * height_m)

# 计算计划BMI和实际BMI
df["计划BMI"] = df["计划体重"].apply(lambda x: calculate_bmi(x, height))
df["实际BMI"] = df["实际体重"].apply(lambda x: calculate_bmi(x, height) if pd.notna(x) else None)

# 创建单个图表
fig, ax1 = plt.subplots(figsize=(14, 8))

# 绘制体重变化（在上半部分）
ax1.plot(df["终止日"], df["计划体重"], label="预期", color="#FFA07A", linewidth=2, linestyle='--')  # 浅鲑鱼色
ax1.plot(df["终止日"], df["实际体重"], label="实际", color="#87CEEB", linewidth=2, marker='o')  # 天蓝色

# 标注所有预期体重点
for i, row in df.iterrows():
    plan_weight = row["计划体重"]
    ax1.text(row["终止日"], plan_weight + 0.3,  # 向上偏移0.3个单位
             f'{plan_weight}',
             color='#FFA07A',  # 与预期线相同的颜色
             ha='right',       # 右对齐
             va='bottom',      # 底部对齐
             fontsize=8,       # 稍微小一点的字体
             rotation=45)      # 文字旋转45度，节省水平空间

# 标注实际体重点
for i, row in df.iterrows():
    y = row["实际体重"]
    target = row["计划体重"]
    if pd.notna(y):
        if y <= target:
            ax1.plot(row["终止日"], y, 'o', color='#87CEEB', markersize=8)
            ax1.text(row["终止日"], y - 2, f"{y}",  # 向下偏移2个单位
                    ha='center', fontsize=9, color='#87CEEB')
        else:
            ax1.plot(row["终止日"], y, 'o', color='#87CEEB', markersize=8)
            ax1.text(row["终止日"], y - 2, f"{y}",  # 向下偏移2个单位
                    ha='center', fontsize=9, color='#FF6B6B')

# 设置Y轴刻度
min_weight = target_weight // 10 * 10  # 下限为目标体重向下取整到10的倍数
max_weight = start_weight // 10 * 10 + 5  # 上限为起始体重向上取整到10的倍数

# 设置达成率显示的范围（kg）
rate_range = 40  # 达成率占用40kg的显示范围

# 绘制达成率
negative_rates = [min_weight - (rate_range * (100-r)/100) if r is not None else None for r in cumulative_rates]
# ax1.plot(df["终止日"], negative_rates, label="达成率", color="#98FB98", linewidth=2, marker='^')  # 淡绿色
line_rate = ax1.plot(df["终止日"], negative_rates, label="达成率", color="#2E8B57", linewidth=2, marker='^')

# 在达成率点上添加正负贡献标记
for i, (x, y, contrib) in enumerate(zip(df["终止日"], negative_rates, weekly_contributions)):
    if y is not None and contrib is not None:
        marker_color = '#90EE90' if contrib == 'o' else '#FF6B6B'  # 正贡献用浅绿色，负贡献用浅红色
        ax1.plot(x, y, marker=contrib, color=marker_color, markersize=10, 
                markeredgecolor='white', markeredgewidth=1)

# 在达成率区域添加BMI曲线
bmi_values = df["实际BMI"]
# BMI值反向映射，这样BMI越低越接近min_weight
mapped_bmi = [min_weight - (rate_range * (100-x)/100) if pd.notna(x) else None for x in bmi_values]
ax1.plot(df["终止日"], mapped_bmi, label="BMI", color='#FFA07A', linewidth=2, marker='s')  # 浅鲑鱼色

# 生成以5为间隔的体重刻度
weight_ticks = np.arange(min_weight, max_weight + 5, 5)
# 生成每5%一个刻度的达成率和BMI刻度
rate_ticks = [min_weight - (rate_range * t/100) for t in range(0, 101, 5)]

# 设置所有刻度
ax1.set_yticks(list(weight_ticks) + rate_ticks)

# 设置Y轴范围
ax1.set_ylim(
    min_weight - rate_range,  # 达成率的最小值
    max_weight   # 体重的最大值
)

# 修改格式化函数
def custom_formatter(x, p):
    if x == min_weight:
        return '100'
    elif x < min_weight:
        # 计算百分比值（同时适用于达成率和BMI）
        percent = 100 - ((min_weight - x) / rate_range * 100)
        return f'{percent:.0f}'
    return str(int(x))

ax1.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))

# 添加BMI分类区域
# BMI分类函数
# bmi = start_weight / (height * height)
# BMI 值范围	分类
# < 18.5	体重过轻
# 18.5–24.9	正常体重
# 25–29.9	超重
# 30–34.9	肥胖 I 级
# 35–39.9	肥胖 II 级
# ≥ 40	重度肥胖
bmi_categories = [
    (40, 35, "肥胖_II", '#8B0000'),    # 深红色
    (35, 30, "肥胖_I", '#CD5C5C'),    # 印度红
    (30, 25, "超重", '#B8860B'),     # 暗金色
    (25, 18.5, "正常", '#228B22'),  # 森林绿
    (18.5, 15, "过轻", '#4682B4')   # 钢青色
]

for high, low, category, color in bmi_categories:
  # 反转BMI值的映射
    y_high = min_weight - (rate_range * (100-high)/100)
    y_low = min_weight - (rate_range * (100-low)/100)
    # 填充区域
    ax1.axhspan(y_low, y_high, alpha=0.4, color=color)
    # 添加区域边界线
    ax1.axhline(y=y_high, color='white', alpha=0.3, linewidth=1.5)  # 增加边界线的可见度
    ax1.axhline(y=y_low, color='white', alpha=0.3, linewidth=1.5)   # 增加边界线的可见度
    # 添加文字标签
    ax1.text(df["终止日"].iloc[-1], (y_high + y_low)/2, category, 
             ha='left', va='center', color='grey', alpha=0.7)

# 设置坐标轴
# ax1.set_xlabel("日期")
# ax1.set_ylabel("体重（kg）")
ax1.tick_params(axis='x', rotation=90)

# 添加网格，使用暗色网格
ax1.grid(True, color='gray', alpha=0.3)

# 添加图例
ax1.legend(loc='upper right')

# 设置标题
# ax1.set_title("减重进度", pad=15)

# 调整布局
plt.tight_layout()
plt.show()
