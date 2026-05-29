import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# -------------------------------
# 1. 전역 스타일 및 한글 웹폰트 안전 주입 (CSS 방식)
# -------------------------------
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
    '주변환경': [3, 7, 2, 8, 4, 9, 6, 6],
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
# 7. 예측 및 시각화 구동 (st.scatter_chart 컴포넌트 활용)
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

    # 차트 시각화 데이터 구성
    st.subheader("📈 환자 위치 시각화")
    
    # 기존 데이터 가공
    chart_df = df.copy()
    chart_df['대상'] = chart_df['cluster'].apply(lambda x: f"위험군 {x}")
    
    # 새 환자 데이터 추가
    new_row = pd.DataFrame({
        '술여부': [Alkhol],
        '주변환경': [AreaQ],
        '담배여부': [Smokes],
        'cluster': [prediction[0]],
        '대상': ['새 환자']
    })
    chart_df = pd.concat([chart_df, new_row], ignore_index=True)

    # ⚠️ 절대 에러가 나지 않는 최신 Streamlit 네이티브 인터랙티브 차트 적용
    st.scatter_chart(
        data=chart_df,
        x='담배여부',
        y='술여부',
        color='대상',
        size='대상',  # 새 환자를 차트에서 더 크게 강조하기 위한 트릭
    )
