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
        font-family: 'Kkukkkukk', 'Nanum Gothic Coding', sans-serif !important;
    }
    /* 밀리터리 분위기를 위한 포인트 컬러 및 배경 가이드 (선택 사항) */
    .stButton>button {
        background-color: #4B5320 !important;
        color: white !important;
        font-weight: bold !important;
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
# 3. 예시 데이터 (0~10 정수 기준: 활동성, 충성도, 훈련강도)
# -------------------------------
df = pd.DataFrame({
    '활동성': [8, 2, 9, 1, 7, 3, 6, 4],
    '충성도': [3, 7, 2, 8, 4, 9, 6, 6],
    '훈련강도': [9, 8, 2, 1, 7, 3, 6, 4]
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
# 6. Streamlit UI 화면 구성 (군가나지 컨셉)
# -------------------------------
st.set_page_config(
    page_title="군가나지 보직 예측",
    page_icon="🪖",
    layout="centered"
)

st.title("🪖 군가나지(Goonganaji) 보직 군집 예측 시스템")
st.write("신병 군가나지의 능력치를 입력하십시오! 악!")

# 입력값 변경 (0~10 정수형)
Activity = st.number_input("🐕 활동성 (개풀 뜯어먹는 스피드 0 ~ 10)", min_value=0, max_value=10, value=0, step=1)
Loyalty = st.number_input("🫡 충성도 (간식 앞에서의 복종력 0 ~ 10)", min_value=0, max_value=10, value=0, step=1)
Training = st.number_input("🏋️ 훈련강도 (개껌 씹기 및 구르기 0 ~ 10)", min_value=0, max_value=10, value=0, step=1)

# -------------------------------
# 7. 예측 및 시각화 구동
# -------------------------------
if st.button("전투력 측정 및 보직 배치하기"):

    new_candidate = pd.DataFrame(
        [[Activity, Loyalty, Training]],
        columns=['활동성', '충성도', '훈련강도']
    )

    new_candidate_scaled = scaler.transform(new_candidate)
    prediction = model.predict(new_candidate_scaled)

    st.subheader("📊 배치 결과 보고")
    
    # 군집 결과 컨셉 매핑
    if prediction[0] == 0:
        st.success("💚 [행정형 군가나지] 후방 꿀보직 및 PX 관리 요원으로 임무를 명 받았습니다!")
    else:
        st.error("💥 [최정예 수색 전우조] 전방 돌격 최정예 훈련 가나지로 군장에 당첨되었습니다!")

    st.write(f"분류 군집 번호: {prediction[0]}")

    # 차트 시각화 데이터 구성
    st.subheader("📈 부대 내 군가나지 위치 시각화")
    
    # 기존 데이터 가공
    chart_df = df.copy()
    chart_df['분류'] = chart_df['cluster'].apply(lambda x: f"행정 가나지군" if x == 0 else f"전투 가나지군")
    chart_df['크기'] = 150  # 기존 데이터 점 크기
    
    # 신규 캐릭터 데이터 추가
    new_row = pd.DataFrame({
        '활동성': [Activity],
        '충성도': [Loyalty],
        '훈련강도': [Training],
        'cluster': [prediction[0]],
        '분류': ['★ 신병 군가나지 (현재 입력값)'],
        '크기': [450]  # 신병 캐릭터를 차트에서 독보적으로 크게 강조하기 위한 크기 트릭
    })
    chart_df = pd.concat([chart_df, new_row], ignore_index=True)

    # 에러 없는 Streamlit 네이티브 차트로 군가나지 위치 매핑
    st.scatter_chart(
        data=chart_df,
        x='훈련강도',
        y='활동성',
        color='분류',
        size='크기'
    )
