import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# === Step 1: 数据准备 ===
df1 = pd.read_excel("1.xlsx")  # 替换为你的文件名
df2 = pd.read_excel("2.xlsx")

def classify_salt_content(salt):
    if salt < 0.1:
        return 0
    elif salt < 0.25:
        return 1
    elif salt < 0.5:
        return 2
    else:
        return 3

df1 = df1.drop(columns=["Unnamed: 0"], errors="ignore")
df2 = df2.drop(columns=["Unnamed: 0"], errors="ignore")
df1["Salt_Level"] = df1["Total salt content, %"].apply(classify_salt_content)
df2["Salt_Level"] = df2["Total salt content, %"].apply(classify_salt_content)

features = ['HCO₃⁻', 'CO₃²⁻', 'Cl⁻', 'SO₄²⁻', 'Ca²⁺', 'Mg²⁺', 'Na⁺', 'K⁺']
X1, y1 = df1[features], df1["Salt_Level"]
X2, y2 = df2[features], df2["Salt_Level"]

# === Step 2: 模型训练 ===
# Region_1 → Region_1
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model_1 = RandomForestClassifier(random_state=42)
model_1.fit(X1_train, y1_train)
y1_pred = model_1.predict(X1_test)
report_1 = classification_report(y1_test, y1_pred)
print(" Region_1 → Region_1\n", report_1)

# Region_2 → Region_2
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model_2 = RandomForestClassifier(random_state=42)
model_2.fit(X2_train, y2_train)
y2_pred = model_2.predict(X2_test)
report_2 = classification_report(y2_test, y2_pred)
print(" Region_2 → Region_2\n", report_2)

# Cross-prediction
y_cross_1to2 = model_1.predict(X2)
y_cross_2to1 = model_2.predict(X1)

report_cross_1to2 = classification_report(y2, y_cross_1to2, output_dict=True)
report_cross_2to1 = classification_report(y1, y_cross_2to1, output_dict=True)

# === Step 3: 准确率柱状图 ===
acc_results = {
    "Region_1 → Region_1": 1.00,
    "Region_2 → Region_2": 0.913,
    "Region_1 → Region_2": report_cross_1to2["accuracy"],
    "Region_2 → Region_1": report_cross_2to1["accuracy"]
}

plt.figure(figsize=(8, 5))
sns.barplot(x=list(acc_results.keys()), y=list(acc_results.values()))
plt.ylim(0, 1.1)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xlabel("Training → Testing Region")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("accuracy_comparison.png", dpi=300)
plt.show()

# === Step 4: 特征重要性图 & 保存 ===
importance_df = pd.DataFrame({
    'Feature': features,
    'Region_1': model_1.feature_importances_,
    'Region_2': model_2.feature_importances_
})
importance_df.set_index("Feature").plot(kind="bar", figsize=(10, 6))
plt.title("Feature Importance Comparison by Region")
plt.ylabel("Importance Score")
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("feature_importance_comparison.png", dpi=300)
plt.show()

importance_df.set_index("Feature").to_csv("feature_importance_comparison.csv")

# === Step 5: 混淆矩阵图保存函数 ===
def plot_confusion_matrix(cm, title, filename):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Non", "Slight", "Moderate", "Severe"],
                yticklabels=["Non", "Slight", "Moderate", "Severe"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()

# 混淆矩阵并保存
plot_confusion_matrix(confusion_matrix(y1_test, y1_pred), "CM: Region_1 → Region_1", "cm_region1_train_test.png")
plot_confusion_matrix(confusion_matrix(y2_test, y2_pred), "CM: Region_2 → Region_2", "cm_region2_train_test.png")
plot_confusion_matrix(confusion_matrix(y2, y_cross_1to2), "CM: Region_1 → Region_2", "cm_region1_to_region2.png")
plot_confusion_matrix(confusion_matrix(y1, y_cross_2to1), "CM: Region_2 → Region_1", "cm_region2_to_region1.png")
