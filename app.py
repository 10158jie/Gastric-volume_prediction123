import streamlit as st
import numpy as np
import joblib

# 加载模型（确保和 app.py 在同一目录）
model = joblib.load("adaboost_model1.pkl")
# 页面标题
st.title("Gastric Volume Prediction Tool (AdaBoost Model)")
st.subheader("Please enter the following parameters:")

# 用户输入
age = st.number_input("Age（years）", min_value=18, max_value=100, value=40)
rld_csa = st.number_input("RLD.CSA（cm²）", min_value=1.0, max_value=30.0, value=12.0, step=0.1)
perlas_grade = st.selectbox("Perlas grade", options=["0", "1", "2"])
weight = st.number_input("Weight（kg）", min_value=30.0, max_value=150.0, value=60.0, step=0.5)

# 将Perlas分级转换为one-hot编码
perlas0 = 1 if perlas_grade == "0" else 0
perlas1 = 1 if perlas_grade == "1" else 0
perlas2 = 1 if perlas_grade == "2" else 0

# 执行预测
if st.button("Predicted gastric volume"):
    input_features = np.array([[perlas0, perlas1, perlas2, rld_csa, age]])
    pred_scaled = model.predict(input_features)[0]
    pred_volume = pred_scaled * 100  # 还原为 mL

    st.success(f"Predicted gastric volume：{pred_volume:.1f} mL")

    # 个体化风险分层阈值
    medium_high_threshold = max(50, 0.8 * weight)
    high_threshold = max(100, 1.5 * weight)

    # 风险提示
    if pred_volume >= high_threshold:
        st.error("🚨 High aspiration risk（≥100 mL 或 ≥1.5 mL/kg）")
    elif pred_volume >= medium_high_threshold:
        st.warning("⚠️ Medium-high aspiration risk（≥50 mL 或 ≥0.8 mL/kg）")
    else:
        st.info("✅ Low aspiration risk")
        
