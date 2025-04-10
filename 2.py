import joblib
import pandas as pd

# === 加载训练好的模型 ===
model_region1 = joblib.load("model_region1.pkl")  # 加载 Region_1 模型
model_region2 = joblib.load("model_region2.pkl")  # 加载 Region_2 模型


# === 土壤数据输入和预测 ===
def classify_soil(sample_data, region='Region_1'):
    """
    对土壤样本进行退化分类
    :param sample_data: 一个字典，包含各个离子浓度的数据
    :param region: 'Region_1' 或 'Region_2'，根据区域选择对应模型
    :return: 预测的退化等级
    """
    # 假设输入样本为字典，例如：{'HCO₃⁻': 0.01, 'Cl⁻': 0.03, ...}
    sample_df = pd.DataFrame([sample_data])

    # 输入特征
    features = ['HCO₃⁻', 'CO₃²⁻', 'Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Na⁺', 'K⁺']
    sample_df = sample_df[features]  # 提取相关特征

    # 根据选择的区域加载模型
    if region == 'Region_1':
        model = model_region1
    elif region == 'Region_2':
        model = model_region2
    else:
        raise ValueError("Region must be 'Region_1' or 'Region_2'")

    # 预测
    prediction = model.predict(sample_df)

    # 返回预测结果
    return prediction[0]


# === 示例：输入一个土壤样本 ===
sample_data = {
    'HCO₃⁻': 0.02,  # 示例输入数据
    'CO₃²⁻': 0.0,
    'Cl⁻': 0.05,
    'SO₄²⁻': 0.15,
    'Ca²⁺': 0.02,
    'Mg²⁺': 0.01,
    'Na⁺': 0.1,
    'K⁺': 0.003
}

# 根据 Region_1 模型预测土壤退化等级
predicted_class = classify_soil(sample_data, region='Region_1')

# 输出预测结果
degradation_levels = ['Non-salinized', 'Slight', 'Moderate', 'Severe']
print(f"预测土壤退化等级: {degradation_levels[predicted_class]}")
