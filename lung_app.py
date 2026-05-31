import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -------------------------------
# 1. 한글 폰트 설정 (Matplotlib 깨짐 방지)
# -------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 2. 상대 경로 지정
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "lung_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
data_path = os.path.join(BASE_DIR, "lung.csv")

# -------------------------------
# 3. Streamlit UI 페이지 설정
# -------------------------------
st.set_page_config(page_title="환자 군집 예측 시스템", page_icon="🫁", layout="centered")

st.title("📈 군집 시각화")
st.subheader("폐암 환자 군집 예측 시스템")
st.write("음주량, 주변환경, 흡연량을 입력하면 군집을 예측하고 시각화합니다.")

# 파일들이 정상적으로 존재하는지 체크
if (
    not os.path.exists(model_path)
    or not os.path.exists(scaler_path)
    or not os.path.exists(data_path)
):
    st.error("필수 파일(lung_model.pkl, scaler.pkl, lung.csv)이 깃허브에 없습니다. 확인해 주세요!")
    st.stop()

# -------------------------------
# 4. 모델 및 데이터 로드
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

# 데이터 내부의 실제 컬럼명들을 순서대로 가져옵니다.
cols = df.columns.tolist()

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

    # 💡 [핵심 수정] 컬럼명 불일치 에러를 방지하기 위해 .values를 사용하여 이름 없는 순수 숫자 배열로 변환합니다.
    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]], columns=[cols[0], cols[1], cols[2]]
    )
    
    # 스케일러와 모델에 이름 대신 값(.values)만 밀어 넣습니다.
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    st.success(f"🎯 분석 결과: 이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # -------------------------------
    # 7. 첫 번째 사진 스타일 완벽 복구 (Matplotlib 시각화)
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    # 데이터 내에서 군집을 뜻하는 컬럼 찾기
    cluster_col = None
    for c in df.columns:
        if "cluster" in c.lower() or "군집" in c:
            cluster_col = c
            break

    if cluster_col and cluster_col in df.columns:
        cluster_color = df[cluster_col]
    else:
        cluster_color = "gray"

    # 시각화 데이터 매핑 (X축: 흡연/세번째 열, Y축: 술/첫번째 열)
    ax.scatter(
        df[cols[2]], df[cols[0]], c=cluster_color, cmap="viridis", alpha=0.6, s=60
    )

    # '새 환자' 위치에 검은색 큰 X 표시
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

    # 축 레이블 설정
    ax.set_xlabel("흡연", fontsize=11)
    ax.set_ylabel("알코올", fontsize=11)
    ax.set_title("폐암 환자 군집", fontsize=14, pad=10)
    ax.legend(loc="upper left", fontsize=11)

    # 스트림릿에 그래프 출력
    st.pyplot(fig)
