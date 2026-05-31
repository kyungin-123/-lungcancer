import os
import urllib.request
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# -------------------------------
# 0. ⚠️ [중요] 본인의 깃허브 Raw 주소로 변경하세요!
# -------------------------------
# 깃허브에서 파일을 열고 'Raw' 버튼을 눌렀을 때 나오는 URL 주소 형식을 입력해야 합니다.
GITHUB_RAW_URL = "https://raw.githubusercontent.com/본인ID/저장소이름/main/"

# -------------------------------
# 1. 한글 폰트 설정 (Matplotlib 깨짐 방지)
# -------------------------------
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

# -------------------------------
# 2. 파일 다운로드 함수 (깃허브 -> 로컬 자동 저장)
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def download_file_from_github(filename):
    local_path = os.path.join(BASE_DIR, filename)
    # 파일이 없을 때만 깃허브에서 다운로드 진행
    if not os.path.exists(local_path):
        with st.spinner(f"깃허브에서 {filename} 파일을 다운로드 중입니다..."):
            url = GITHUB_RAW_URL + filename
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                st.error(
                    f"{filename} 다운로드 실패! 깃허브 주소나 파일명을 확인해주세요. 에러: {e}"
                )
                st.stop()
    return local_path


# 깃허브에서 세 가지 파일 자동 원격 다운로드
model_path = download_file_from_github("lung_model.pkl")
scaler_path = download_file_from_github("scaler.pkl")
data_path = download_file_from_github("patient_data.csv")

# -------------------------------
# 3. Streamlit UI 페이지 설정
# -------------------------------
st.set_page_config(page_title="환자 군집 예측 시스템", page_icon="🫁", layout="centered")

st.title("📈 군집 시각화")
st.subheader("폐암 환자 군집 예측 시스템")
st.write("음주량, 주변환경, 흡연량을 입력하면 군집을 예측하고 시각화합니다.")

# -------------------------------
# 4. 모델 및 데이터 로드
# -------------------------------
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
df = pd.read_csv(data_path)

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

    new_patient = pd.DataFrame(
        [[Alkhol, AreaQ, Smokes]], columns=["술여부", "주변환경", "담배여부"]
    )

    new_patient_scaled = scaler.transform(new_patient)
    pred_cluster = model.predict(new_patient_scaled)

    st.success(f"🎯 분석 결과: 이 환자는 {pred_cluster[0]}번 군집에 속합니다.")

    # -------------------------------
    # 7. Matplotlib 시각화
    # -------------------------------
    fig, ax = plt.subplots(figsize=(7, 6))

    if "cluster" in df.columns:
        cluster_color = df["cluster"]
    else:
        cluster_color = "gray"

    scatter = ax.scatter(
        df["담배여부"], df["술여부"], c=cluster_color, cmap="viridis", alpha=0.6, s=60
    )

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

    ax.set_xlabel("흡연", fontsize=11)
    ax.set_ylabel("알코올", fontsize=11)
    ax.set_title("폐암 환자 군집", fontsize=14, pad=10)
    ax.legend(loc="upper left", fontsize=11)

    st.pyplot(fig)
