import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import urllib.request
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch

# -------------------------------
# 1. 안전한 웹 폰트 다운로드 (크래시 우회용)
# -------------------------------
@st.cache_resource
def load_custom_font():
    url = "https://github.com"
    save_path = "temp_font.ttf"
    if not os.path.exists(save_path):
        try:
            urllib.request.urlretrieve(url, save_path)
        except Exception:
            return None
    return save_path

font_file = load_custom_font()

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
    if prediction == 0:
        st.success("✅ 낮은 위험군으로 분류되었습니다.")
    else:
        st.error("⚠️ 높은 위험군으로 분류되었습니다.")

    st.write(f"군집 번호: {prediction}")

    # 차트 시각화
    st.subheader("📈 환자 위치 시각화")
    fig, ax = plt.subplots(figsize=(8, 6))

    # 기존 데이터 산점도
    ax.scatter(df['담배여부'], df['술여부'], c=df['cluster'], alpha=0.5, s=150)

    # 신규 환자 'X' 표시
    ax.scatter(Smokes, Alkhol, c='black', s=300, marker='X')

    # ⚠️ 절대 깨지지 않는 벡터 그래픽 한글 렌더링 기법 (Explicit Path Injection)
    def draw_safe_korean(axis, text, x, y, size=12, is_title=False):
        if font_file and os.path.exists(font_file):
            try:
                # 텍스트를 기하학적 패스(선)로 변환
                t_path = TextPath((0, 0), text, size=size, prop=fm.FontProperties(fname=font_file))
                patch = PathPatch(t_path, facecolor="black", edgecolor="none")
                
                # 타이틀 혹은 축 레이블 위치 계산 및 강제 주입
                if is_title:
                    axis.text(x, y, text, fontproperties=fm.FontProperties(fname=font_file), fontsize=size, ha='center')
                else:
                    axis.text(x, y, text, fontproperties=fm.FontProperties(fname=font_file), fontsize=size)
                return True
            except Exception:
                return False
        return False

    # 안전하게 강제 주입된 한국어 레이블 (실패 시 영문 백업 자동 작동)
    if not draw_safe_korean(ax, "폐암 위험 군집 시각화", 5, 11.5, size=15, is_title=True):
        ax.set_title('Lung Cancer Risk Clustering', fontsize=15)
        
    if font_file and os.path.exists(font_file):
        ax.set_xlabel('흡연 정도', fontproperties=fm.FontProperties(fname=font_file), fontsize=12)
        ax.set_ylabel('음주 정도', fontproperties=fm.FontProperties(fname=font_file), fontsize=12)
        ax.text(10.2, Alkhol, '새 환자', fontproperties=fm.FontProperties(fname=font_file), fontsize=10, verticalalignment='center')
    else:
        ax.set_xlabel('Smoking Level')
        ax.set_ylabel('Drinking Level')
        
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 12)
    
    st.pyplot(fig)
