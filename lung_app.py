import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import streamlit as st

# -------------------------------
# 1. 💡 [한글 패치 핵심] 깃허브 내부 폰트 파일 연결
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(BASE_DIR, "Kkukkukk.ttf")

if os.path.exists(font_path):
    # 업로드되어 있는 Kkukkukk.ttf 폰트를 등록합니다.
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
else:
    # 폰트 파일이 없을 경우 기본 맑은 고딕 시도
    plt.rcParams["font.family"] = "Malgun Gothic"

plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 2. 상대 경로 지정
# -------------------------------
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

# 파일 존재 체크
if (
    not os.path.exists(model_path)
    or not os.path.exists(scaler_path)
    or not os.path.exists(data_path)
):
    st.error("필수 파일이 깃허브에 없습니다. 확인해 주세요!")
    st.stop()

# -------------------------------
# 4. 모델 및 데이터 로드
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

cols = df.columns.tolist()

# -------------------------------
# 5. 사용자 입력
# -------------------------------
Alkhol = st.number_input("음주량 입력 (알코올)", min_value=0.0, step=0.1, value=0.0)
AreaQ = st.number_input("주변환경 입력", min_value=0.0, step=0.1, value=0.0)
Smokes = st.number_input("담배량 입력 (흡연)", min_value=0.0, step=0.1, value=0.0)

# -------------------------------
# 6. 예측 및 시각화 실행
# -------------------------------
if st.button("군집 예측 및 시각화하기"):

    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]], columns=[cols[0], cols[1], cols[2]]
    )
    
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    st.success(f"🎯 분석 결과: 이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # -------------------------------
    # 7. Matplotlib 시각화 (한글 폰트 지정 형식)
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    cluster_col = None
    for c in df.columns:
        if "cluster" in c.lower() or "군집" in c:
            cluster_col = c
            break

    if cluster_col and cluster_col in df.columns:
        cluster_color = df[cluster_col]
    else:
        cluster_color = "gray"

    # 산점도 그래프 그리기 (Y축: 첫번째 컬럼, X축: 세번째 컬럼)
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

    # 💡 [중요] fontproperties 속성을 사용하여 한글을 강제로 먹여줍니다.
    if os.path.exists(font_path):
        ax.set_xlabel("흡연", fontsize=12, fontproperties=font_prop)
        ax.set_ylabel("알코올", fontsize=12, fontproperties=font_prop)
        ax.set_title("폐암 환자 군집", fontsize=15, pad=10, fontproperties=font_prop)
        ax.legend(loc="upper left", prop=font_prop)
    else:
        ax.set_xlabel("흡연", fontsize=12)
        ax.set_ylabel("알코올", fontsize=12)
        ax.set_title("폐암 환자 군집", fontsize=15, pad=10)
        ax.legend(loc="upper left")

    st.pyplot(fig)
