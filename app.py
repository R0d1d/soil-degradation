from flask import Flask, render_template, request
import joblib
import pandas as pd

# 加载训练好的模型
model_region1 = joblib.load("model_region1.pkl")
model_region2 = joblib.load("model_region2.pkl")

# 创建 Flask 应用
app = Flask(__name__)


# 定义土壤数据预测函数
def classify_soil(sample_data, region='Region_1'):
    """
    对土壤样本进行退化分类
    :param sample_data: 一个字典，包含各个离子浓度的数据
    :param region: 'Region_1' 或 'Region_2'，根据区域选择对应模型
    :return: 预测的退化等级
    """
    features = ['HCO₃⁻', 'CO₃²⁻', 'Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Na⁺', 'K⁺']
    sample_df = pd.DataFrame([sample_data])
    sample_df = sample_df[features]

    if region == 'Region_1':
        model = model_region1
    elif region == 'Region_2':
        model = model_region2
    else:
        raise ValueError("Region must be 'Region_1' or 'Region_2'")

    prediction = model.predict(sample_df)
    return prediction[0]


# 路由：首页展示表单
@app.route('/')
def home():
    return render_template('index.html')


# 路由：预测页面
@app.route('/predict', methods=['POST'])
def predict():
    # 获取输入数据
    sample_data = {
        'HCO₃⁻': float(request.form['HCO₃⁻']),
        'CO₃²⁻': float(request.form['CO₃²⁻']),
        'Cl⁻': float(request.form['Cl⁻']),
        'SO₄²⁻': float(request.form['SO₄²⁻']),
        'Ca²⁺': float(request.form['Ca²⁺']),
        'Mg²⁺': float(request.form['Mg²⁺']),
        'Na⁺': float(request.form['Na⁺']),
        'K⁺': float(request.form['K⁺'])
    }

    # 获取区域选项
    region = request.form['region']

    # 进行预测
    predicted_class = classify_soil(sample_data, region)
    degradation_levels = ['Non-salinized', 'Slight', 'Moderate', 'Severe']
    result = degradation_levels[predicted_class]

    return render_template('index.html', prediction=result)


# 运行应用
if __name__ == "__main__":
    app.run(debug=True)
