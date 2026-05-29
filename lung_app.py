import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request

# -------------------------------
# 웹 서버(Streamlit Cloud) 한글 폰트 다운로드 및 설정
# -------------------------------
@st.cache_resource
def load_font():
    font_url = "https://github.com"
    font_path = "NanumGothic.ttf"
    if not os.path.exists(font_path):
        urllib.request.urlretrieve(font_url, font_path)
    return font_path

try:
    font_p = load_font()
    font_name = fm.FontProperties(fname=font_p).get_name()
    plt.rcParams['font.family'] = font_name
except Exception as e:
    # 로컬 실행 대비 예외 처리
    plt.rcParams['font.family'] = 'Malgun Gothic'

plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 모델 & 스케일러 경로
# -------------------------------
model_path = r"C:\streamlit\kmeans_model.pkl"
scaler_path = r"C:\streamlit\scaler.pkl"

# -------------------------------
# 예시 데이터 (0~10 정수 스케일에 맞춤)
# -------------------------------
df = pd.DataFrame({
    '술여부': [8, 2, 9, 1, 7, 3, 6, 4],
    '주변환경': [3, 7, 2, 8, 4, 9, 5, 6],
    '담배여부': [9, 8, 2, 1, 7, 3, 6, 4]
})

# -------------------------------
# 모델 없으면 자동 생성
# -------------------------------
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    # 스케일러 학습 및 적용
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # KMeans 모델 학습
    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X_scaled)

    # 파일 저장
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

# -------------------------------
# 모델 및 스케일러 불러오기
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# 기존 데이터 군집 반영
X_scaled = scaler.transform(df)
df['cluster'] = model.predict(X_scaled)

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(
    page_title="폐암 위험도 예측",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 폐암 위험도 군집 예측 시스템")
st.write("환자 정보를 입력하세요.")

# 입력값 (0~10 범위의 정수로 변경)
Alkhol = st.number_input(
    "🍺 음주 정도 (0 ~ 10)",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

AreaQ = st.number_input(
    "🏭 주변 환경 위험도 (0 ~ 10)",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)

Smokes = st.number_input(
    "🚬 흡연 정도 (0 ~ 10)",
    min_value=0,
    max_value=10,
    value=0,
    step=1
)
# -------------------------------
# 예측 버튼
# -------------------------------
if st.button("예측하기"):
    # 새로운 환자 데이터
    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]],
        columns=['술여부', '주변환경', '담배여부']
    )

    # 스케일링 및 예측
    new_patient_scaled = scaler.transform(new_patient)
    prediction = model.predict(new_patient_scaled)

    # -------------------------------
    # 결과 출력
    # -------------------------------
    st.subheader("📊 예측 결과")
    if prediction[0] == 0:
        st.success("✅ 낮은 위험군으로 분류되었습니다.")
    else:
        st.error("⚠️ 높은 위험군으로 분류되었습니다.")

    st.write(f"군집 번호: {prediction[0]}")

    # -------------------------------
    # 시각화
    # -------------------------------
    st.subheader("📈 환자 위치 시각화")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 기존 데이터 표시
    ax.scatter(
        df['담배여부'],
        df['술여부'],
        c=df['cluster'],
        alpha=0.5,
        s=150
    )

    # 새 환자 표시
    ax.scatter(
        Smokes,
        Alkhol,
        c='black',
        s=300,
        marker='X',
        label='새 환자'
    )

    ax.set_xlabel('흡연 정도')
    ax.set_ylabel('음주 정도')
    ax.set_title('폐암 위험 군집 시각화')
    ax.set_xlim(-1, 11)  # 그래프 축 범위 고정
    ax.set_ylim(-1, 11)
    ax.legend()

    st.pyplot(fig)
