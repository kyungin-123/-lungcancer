import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -------------------------------
# 한글 폰트 설정 (Matplotlib 깨짐 방지)
# -------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"  # 윈도우 기준 (맥은 'AppleGothic')
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 경로 설정 (절대 경로로 파일 로드 오류 방지)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "lung_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
data_path = os.path.join(BASE_DIR, "patient_data.csv")

# -------------------------------
# 제목
# -------------------------------
st.title("환자 군집 예측 시스템")
st.write("음주량, 주변환경, 흡연량을 입력하면 군집을 예측합니다.")

# -------------------------------
# 모델 및 데이터 불러오기
# -------------------------------
# 파일들이 lung_app.py와 같은 폴더에 있어야 합니다.
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

# -------------------------------
# 사용자 입력
# -------------------------------
Alkhol = st.number_input("음주량 입력", min_value=0.0, step=0.1)
AreaQ = st.number_input("주변환경 입력", min_value=0.0, step=0.1)
Smokes = st.number_input("담배량 입력", min_value=0.0, step=0.1)

# -------------------------------
# 예측 버튼
# -------------------------------
if st.button("군집 예측하기"):

    # 새로운 환자 데이터 생성 (모델 학습 시 사용한 컬럼명과 일치해야 합니다)
    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]], columns=["술여부", "주변환경", "담배여부"]
    )

    # 스케일링 및 군집 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 출력
    st.success(f"이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # -------------------------------
    # 시각화
    # -------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # 원본 데이터에 'cluster' 컬럼이 없다면 방금 예측한 값으로 임시 시각화하거나 제거해야 합니다.
    # 여기서는 기존 코드 구조를 유지하되 안전하게 색상을 지정합니다.
    if "cluster" in df.columns:
        c_target = df["cluster"]
    else:
        c_target = "blue"  # cluster 컬럼이 없을 경우 기본 색상

    scatter = ax.scatter(df["담배여부"], df["술여부"], c=c_target, alpha=0.5)

    # 새 환자 표시 (X 표시)
    ax.scatter(Smokes, Alkhol, c="black", s=300, marker="X", label="새 환자")

    ax.set_xlabel("흡연")
    ax.set_ylabel("음주")
    ax.set_title("환자 군집 시각화")
    ax.legend()

    # 스트림릿에 그래프 출력
    st.pyplot(fig)
