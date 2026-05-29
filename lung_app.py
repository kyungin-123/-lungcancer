import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# -------------------------------
# 1. 폰트 강제 등록 및 한글 설정
# -------------------------------
font_path = "Kkukkkuk.ttf"  # 깃허브에 업로드한 폰트 파일명

if os.path.exists(font_path):
    try:
        # 매트플롯립에 폰트 직접 주입
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rcParams['font.family'] = font_name
    except Exception as e:
        plt.rcParams['font.family'] = 'sans-serif'
else:
    # 폰트 파일이 없을 경우 경고 메시지 출력
    st.sidebar.warning(f"⚠️ 폰트 파일('{font_path}')을 찾을 수 없어 기본 폰트로 대체합니다.")
    plt.rcParams['font.family'] = 'sans-serif'

plt.rcParams['axes.unicode_minus'] = False

# -------------------------------
# 2. 모델 & 스케일러 경로 (상대 경로로 수정)
# -------------------------------
# 웹 서버와 로컬 모두에서 동작하도록 상대 경로 지정
model_path = "kmeans_model.pkl"
scaler_path = "scaler.pkl"

# -------------------------------
# 3. 예시 데이터 (0~10 정수 기준 수정)
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

# 기존 데이터 군집 컬럼 생성
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

# 입력값 (0 ~ 10 정수형태 세팅)
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
# 7. 예측 및 시각화 구동
# -------------------------------
if st.button("예측하기"):

    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]],
        columns=['술여부', '주변환경', '담배여부']
    )

    new_patient_scaled = scaler.transform(new_patient)
    prediction = model.predict(new_patient_scaled)

    # 결과 출력
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
    ax.scatter(
        df['담배여부'],
        df['술여부'],
        c=df['cluster'],
        alpha=0.5,
        s=150
    )

    # 신규 환자 'X' 표시
    ax.scatter(
        Smokes,
        Alkhol,
        c='black',
        s=300,
        marker='X',
        label='새 환자'
    )

    # 축 설정 및 타이틀 (등록한 폰트로 한글 출력 가능)
    ax.set_xlabel('흡연 정도')
    ax.set_ylabel('음주 정도')
    ax.set_title('폐암 위험 군집 시각화')
    
    # 0~10 범위를 예쁘게 보여주기 위한 축 한계 설정
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 11)
    
    ax.legend()
    st.pyplot(fig)
