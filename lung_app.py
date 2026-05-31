import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -------------------------------
# 1. 한글 폰트 설정 (Matplotlib 깨짐 방지)
# -------------------------------
# 윈도우 환경에서 한글이 깨지지 않도록 설정합니다.
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 2. 파일 경로 설정 (절대 경로)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "lung_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
data_path = os.path.join(BASE_DIR, "patient_data.csv")

# -------------------------------
# 3. Streamlit UI 페이지 설정
# -------------------------------
st.set_page_config(page_title="환자 군집 예측 시스템", page_icon="🫁", layout="centered")

st.title("📈 군집 시각화")
st.subheader("폐암 환자 군집 예측 시스템")
st.write("음주량, 주변환경, 흡연량을 입력하면 군집을 예측하고 시각화합니다.")

# -------------------------------
# 4. 모델 및 데이터 로드
# -------------------------------
# 파일이 없으면 에러 메시지를 띄웁니다.
if (
    not os.path.exists(model_path)
    or not os.path.exists(scaler_path)
    or not os.path.exists(data_path)
):
    st.error(
        "필수 파일(lung_model.pkl, scaler.pkl, patient_data.csv)이 같은 폴더에 없습니다. 확인해 주세요!"
    )
    st.stop()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

# -------------------------------
# 5. 사용자 입력 (환자 데이터)
# -------------------------------
Alkhol = st.number_input("음주량 입력 (알코올)", min_value=0.0, step=0.1, value=0.0)
AreaQ = st.number_input("주변환경 입력", min_value=0.0, step=0.1, value=0.0)
Smokes = st.number_input("담배량 입력 (흡연)", min_value=0.0, step=0.1, value=0.0)

# -------------------------------
# 6. 예측 및 시각화 실행
# -------------------------------
if st.button("군집 예측 및 시각화하기"):

    # 새로운 환자 데이터 DataFrame 생성
    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]], columns=["술여부", "주변환경", "담배여부"]
    )

    # 스케일링 및 예측
    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)

    # 결과 메시지 출력
    st.success(f"🎯 분석 결과: 이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # -------------------------------
    # 7. 첫 번째 사진 스타일로 Matplotlib 시각화
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    # 기존 환자 데이터 산점도 (cluster별로 색상 자동 지정)
    # 이미지처럼 'cluster' 컬럼을 기준으로 색을 칠합니다.
    if "cluster" in df.columns:
        cluster_color = df["cluster"]
    else:
        cluster_color = "gray"  # 혹시 데이터에 cluster가 없다면 기본색 지정

    scatter = ax.scatter(
        df["담배여부"],
        df["술여부"],
        c=cluster_color,
        cmap="viridis",
        alpha=0.6,
        s=60,
    )

    # 첫 번째 사진과 똑같이 새 환자를 '검은색 커다란 X' 모양으로 강조
    ax.scatter(
        Smokes,
        Alkhol,
        c="black",
        s=350,
        marker="X",
        edgecolors="white",
        linewidths=1.5,
        label="새 환자",
    )

    # 라벨 및 타이틀 설정 (첫 번째 사진 기준)
    ax.set_xlabel("흡연", fontsize=11)
    ax.set_ylabel("알코올", fontsize=11)
    ax.set_title("폐암 환자 군집", fontsize=14, pad=10)

    # 범례 표시
    ax.legend(loc="upper left", fontsize=11)

    # 스트림릿 화면에 그래프 뿌리기
    st.pyplot(fig)
