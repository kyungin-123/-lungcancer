import os
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import streamlit as st

# -------------------------------
# 1. 폰트 설정 (Kkukkukk.ttf)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
font_path = os.path.join(BASE_DIR, "Kkukkukk.ttf")

if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
else:
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
if not os.path.exists(model_path) or not os.path.exists(scaler_path) or not os.path.exists(data_path):
    st.error("필수 파일이 깃허브에 없습니다. 확인해 주세요!")
    st.stop()

# -------------------------------
# 4. 모델 및 데이터 로드
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

# 💡 [핵심 수정] 데이터에서 실제 분석에 필요한 컬럼을 이름 기반으로 정확하게 찾아냅니다.
# 데이터 내부에 '술', '음주', 'Alkhol', '담배', '흡연', 'Smokes', '주변', 'Area' 등이 포함된 열을 검색합니다.
x_col, y_col, area_col, cluster_col = None, None, None, None

for c in df.columns:
    c_lower = c.lower()
    if "담배" in c or "흡연" in c or "smoke" in c_lower:
        x_col = c  # X축: 흡연
    elif "술" in c or "음주" in c or "alk" in c_lower:
        y_col = c  # Y축: 음주
    elif "주변" in c or "환경" in c or "area" in c_lower:
        area_col = c  # 주변환경
    elif "cluster" in c_lower or "군집" in c:
        cluster_col = c  # 군집 결과 정보

# 만약 이름을 못 찾았다면, 이름 열을 건너뛰고 숫자 데이터가 시작되는 안전한 기본 인덱스로 매칭합니다.
if not x_col: x_col = df.columns[3] if len(df.columns) > 3 else df.columns[2]
if not y_col: y_col = df.columns[1]
if not area_col: area_col = df.columns[2]

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

    # 모델이 학습할 때 기억하던 Feature 컬럼의 순서와 이름을 완벽히 강제 일치시킵니다.
    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]], columns=[y_col, area_col, x_col]
    )
    
    # 안전하게 값만 스케일러와 모델에 주입합니다.
    new_patient_scaled = scaler.transform(new_patient.values)
    pred_cluster = model.predict(new_patient_scaled)

    st.success(f"🎯 분석 결과: 이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # -------------------------------
    # 7. Matplotlib 시각화 (Y축 이름 쏠림 완벽 해결)
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    if cluster_col and cluster_col in df.columns:
        cluster_color = df[cluster_col]
    else:
        cluster_color = "gray"

    # 💡 [해결] 환자 이름이 아닌, 정확히 추출해낸 흡연(x_col)과 알코올(y_col) 숫자 데이터로 그래프를 그립니다!
    ax.scatter(
        df[x_col], df[y_col], c=cluster_color, cmap="viridis", alpha=0.6, s=60
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

    # 한글 라벨 속성 부여
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
