import os
import base64
import joblib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import pandas as pd
import streamlit as st

# -------------------------------
# 1. 기본 경로
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

font_path = os.path.join(BASE_DIR, "Kkukkukk.ttf")
dog_path = os.path.join(BASE_DIR, "dog.png")

# -------------------------------
# 2. 폰트 설정
# -------------------------------
if os.path.exists(font_path):
    font_prop = fm.FontProperties(fname=font_path)
    font_name = font_prop.get_name()
    plt.rcParams["font.family"] = font_name
else:
    plt.rcParams["font.family"] = "Malgun Gothic"

plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 3. 페이지 설정
# -------------------------------
st.set_page_config(
    page_title="폐암 환자 군집 예측",
    page_icon="🫁",
    layout="centered"
)

# -------------------------------
# 4. 배경 이미지 적용
# -------------------------------
def get_base64(file_path):
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

if os.path.exists(dog_path):

    dog_base64 = get_base64(dog_path)

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-color: #f5f5f5;
            background-image: url("data:image/png;base64,{dog_base64}");
            background-repeat: no-repeat;
            background-position: right bottom;
            background-size: 280px;
        }}

        h1 {{
            text-align:center;
        }}

        .stButton > button {{
            width:100%;
            height:55px;
            border-radius:20px;
            font-size:18px;
            font-weight:bold;
            border:2px solid black;
            background:white;
            transition:0.3s;
        }}

        .stButton > button:hover {{
            transform:scale(1.03);
        }}

        [data-testid="stNumberInput"] {{
            background:white;
            border-radius:15px;
            padding:8px;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

# -------------------------------
# 5. 모델 경로
# -------------------------------
model_path = os.path.join(BASE_DIR, "lung_model.pkl")
scaler_path = os.path.join(BASE_DIR, "scaler.pkl")
data_path = os.path.join(BASE_DIR, "lung.csv")

# -------------------------------
# 6. 제목
# -------------------------------
st.markdown(
    "<h1>🫁 폐암 환자 군집 예측 시스템</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>음주량, 주변환경, 흡연량을 입력하면 군집을 예측합니다.</p>",
    unsafe_allow_html=True
)

# -------------------------------
# 7. 파일 존재 확인
# -------------------------------
if not os.path.exists(model_path):
    st.error("lung_model.pkl 파일이 없습니다.")
    st.stop()

if not os.path.exists(scaler_path):
    st.error("scaler.pkl 파일이 없습니다.")
    st.stop()

if not os.path.exists(data_path):
    st.error("lung.csv 파일이 없습니다.")
    st.stop()

# -------------------------------
# 8. 모델 불러오기
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

# -------------------------------
# 9. 컬럼 자동 탐색
# -------------------------------
x_col = None
y_col = None
area_col = None
cluster_col = None

for c in df.columns:

    c_lower = c.lower()

    if "담배" in c or "흡연" in c or "smoke" in c_lower:
        x_col = c

    elif "술" in c or "음주" in c or "alk" in c_lower:
        y_col = c

    elif "주변" in c or "환경" in c or "area" in c_lower:
        area_col = c

    elif "cluster" in c_lower or "군집" in c:
        cluster_col = c

if not x_col:
    x_col = df.columns[3] if len(df.columns) > 3 else df.columns[2]

if not y_col:
    y_col = df.columns[1]

if not area_col:
    area_col = df.columns[2]

# -------------------------------
# 10. 사용자 입력
# -------------------------------
Alkhol = st.number_input(
    "🍺 음주량 입력",
    min_value=0.0,
    step=0.1,
    value=0.0
)

AreaQ = st.number_input(
    "🏠 주변환경 입력",
    min_value=0.0,
    step=0.1,
    value=0.0
)

Smokes = st.number_input(
    "🚬 흡연량 입력",
    min_value=0.0,
    step=0.1,
    value=0.0
)

# -------------------------------
# 11. 예측
# -------------------------------
if st.button("🔍 군집 예측 및 시각화하기"):

    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]],
        columns=[y_col, area_col, x_col]
    )

    new_patient_scaled = scaler.transform(
        new_patient.values
    )

    pred_cluster = model.predict(
        new_patient_scaled
    )

    st.success(
        f"🎯 분석 결과 : {pred_cluster[0]}번 군집입니다."
    )

    # ---------------------------
    # 그래프
    # ---------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    if cluster_col and cluster_col in df.columns:
        cluster_color = df[cluster_col]
    else:
        cluster_color = "gray"

    ax.scatter(
        df[x_col],
        df[y_col],
        c=cluster_color,
        cmap="viridis",
        alpha=0.6,
        s=60
    )

    ax.scatter(
        Smokes,
        Alkhol,
        c="black",
        s=350,
        marker="X",
        edgecolors="white",
        linewidths=1.5,
        label="새 환자"
    )

    if os.path.exists(font_path):

        ax.set_xlabel(
            "흡연",
            fontsize=12,
            fontproperties=font_prop
        )

        ax.set_ylabel(
            "알코올",
            fontsize=12,
            fontproperties=font_prop
        )

        ax.set_title(
            "폐암 환자 군집",
            fontsize=15,
            fontproperties=font_prop
        )

        ax.legend(prop=font_prop)

    else:

        ax.set_xlabel("흡연")
        ax.set_ylabel("알코올")
        ax.set_title("폐암 환자 군집")
        ax.legend()

    st.pyplot(fig)
