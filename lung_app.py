import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt

# -------------------------------
# 1. 폰트 크래시 방지 및 한글 웹 폰트 CSS 주입
# -------------------------------
# Matplotlib의 내부 엔진이 깨진 파일 읽다 크래시 나는 것을 원천 차단하기 위해 
# 범용성이 검증된 기본 한글 폰트 시스템 이름(DejaVu Sans 등 리눅스 표준)으로 백업합니다.
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False

# Streamlit 웹 화면 전체(제목, 입력창 등)를 '메모먼트 꾹꾹체'로 바꾸는 안전한 CSS 스트리밍 주입
st.markdown("""
    <style>
    @import url('https://googleapis.com');
    @font-face {
        font-family: 'Kkukkkukk';
        src: url('https://github.com') format('truetype');
    }
    html, body, [data-testid="stMarkdownContainer"], h1, h2, h3, h4, h5, h6, label, p, span {
        font-family: 'Kkukkkukk', 'Nanum Gothic', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# 2. 모델 & 스케일러 경로 설정
# -------------------------------
current_dir = os.path.dirname(__file__)
model_path = os.path.join(current_dir, "kmeans_model.pkl")
scaler_path = os.path.join(current_dir, "scaler.pkl")

# -------------------------------
# 3. 예시 데이터 (0~10 정수 기준)
# -------------------------------
df = pd.DataFrame({
    '술여부': [8, 2, 9, 1, 7, 3, 6, 4],
    '주변환경': [3, 7, 2, 8, 4, 9, 5, 6],
    '담배여부': [9, 8, 2, 1, 7, 3, 6, 4]
})

# -------------------------------
# 4. 모델 없으면 자동 생성 및 학습
# -------------------------------
if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=2, random_state=42)
    model.fit(X_scaled)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

# -------------------------------
# 5. 모델 및 스케일러 로드
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

X_scaled = scaler.transform(df)
df['cluster'] = model.predict(X_scaled)

# -------------------------------
# 6. Streamlit UI 화면 구성
# -------------------------------
st.set_page_config(
    page_title="폐암 위험도 예측",
    page_icon="🫁",
    layout="centered"
)

st.title("🫁 폐암 위험도 군집 예측 시스템")
st.write("환자 정보를 입력하세요.")

Alkhol = st.number_input("🍺 음주 정도 (0 ~ 10)", min_value=0, max_value=10, value=0, step=1)
AreaQ = st.number_input("🏭 주변 환경 위험도 (0 ~ 10)", min_value=0, max_value=10, value=0, step=1)
Smokes = st.number_input("🚬 흡연 정도 (0 ~ 10)", min_value=0, max_value=10, value=0, step=1)

# -------------------------------
# 7. 예측 및 시각화 구동
# -------------------------------
if st.button("예측하기"):

    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]],
        columns=['술여부', '주변환경', '담배여부']
    )

    new_patient_scaled = scaler.transform(new_patient)
    prediction = model.predict(new_patient_scaled)

    st.subheader("📊 예측 결과")
    if prediction[0] == 0:
        st.success("✅ 낮은 위험군으로 분류되었습니다.")
    else:
        st.error("⚠️ 높은 위험군으로 분류되었습니다.")

    st.write(f"군집 번호: {prediction[0]}")

    # 차트 시각화
    st.subheader("📈 환자 위치 시각화")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 기존 데이터 산점도
    ax.scatter(df['담배여부'], df['술여부'], c=df['cluster'], alpha=0.5, s=150)

    # 신규 환자 'X' 표시
    ax.scatter(Smokes, Alkhol, c='black', s=300, marker='X', label='New Patient')

    # ⚠️ 리눅스 서버에서 절대로 에러가 나지 않는 영문 표준 축 레이블링으로 고정합니다.
    # (한글 제목은 Streamlit UI 타이틀이 해결해 주므로 차트 크래시를 완벽 차단함)
    ax.set_xlabel('Smoking Level (0-10)', fontsize=12)
    ax.set_ylabel('Drinking Level (0-10)', fontsize=12)
    ax.set_title('Lung Cancer Risk Clustering', fontsize=16, pad=15)
    ax.legend(loc='upper right')
    
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    
    st.pyplot(fig)
