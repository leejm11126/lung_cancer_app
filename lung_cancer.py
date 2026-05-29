import streamlit as st
import pandas as pd
import joblib
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager


# ---------------- 페이지 설정 ----------------
st.set_page_config(
    page_title="AI 환자 군집 예측 시스템",
    page_icon="🩺",
    layout="wide"
)

# ---------------- matplotlib 한글 설정 ----------------
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ---------------- CSS 스타일 ----------------
st.markdown("""
<style>

/* 전체 배경 */
.stApp {
    background: linear-gradient(135deg, #0f172a, #111827);
    color: white;
}

/* 메인 카드 */
.main-card {
    background: rgba(255,255,255,0.05);
    padding: 2rem;
    border-radius: 24px;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255,255,255,0.08);
    margin-top: 2rem;
    margin-bottom: 2rem;
}

/* 제목 */
h1, h2, h3 {
    color: white;
}

/* 버튼 */
.stButton > button {
    width: 100%;
    height: 3.2rem;
    border-radius: 16px;
    border: none;
    background: linear-gradient(90deg, #06b6d4, #3b82f6);
    color: white;
    font-size: 1rem;
    font-weight: bold;
    transition: 0.3s;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 25px rgba(59,130,246,0.35);
}

/* metric */
[data-testid="metric-container"] {
    background: rgba(255,255,255,0.05);
    border-radius: 18px;
    padding: 1rem;
    border: 1px solid rgba(255,255,255,0.05);
}

/* dataframe */
[data-testid="stDataFrame"] {
    border-radius: 20px;
    overflow: hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- 모델 로드 ----------------
model = joblib.load("lung_cancer_model.pkl")
scaler = joblib.load("lung_scaler.pkl")

# ---------------- CSV 데이터 로드 ----------------
df_original = pd.read_csv("lung.csv")

# 컬럼명 확인 후 변경
# 실제 CSV 컬럼명에 맞게 수정 가능
df_original.columns = ['Name','Surname','나이','흡연','환경지','음주량','Result']

# ---------------- 메인 카드 ----------------
st.markdown('<div class="main-card">', unsafe_allow_html=True)

st.title("🧬 AI 환자 군집 예측 시스템")

st.caption("""
입력된 건강 데이터를 기반으로
AI가 환자의 군집 위치를 분석합니다.
""")

st.write("")

# ---------------- 입력 영역 ----------------
st.subheader("📋 환자 건강 정보")

col1, col2, col3 = st.columns(3)

with col1:
    input_age = st.slider(
        "나이",
        min_value=0,
        max_value=100,
        value=30
    )

with col2:
    input_smoking = st.slider(
        "흡연 지표",
        min_value=0,
        max_value=50,
        value=5
    )

with col3:
    input_alcohol = st.slider(
        "음주량",
        min_value=0,
        max_value=20,
        value=3
    )

st.write("")

# ---------------- 현재 입력값 표시 ----------------
metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric(
        "나이",
        f"{input_age}세"
    )

with metric2:
    st.metric(
        "흡연",
        f"{input_smoking}"
    )

with metric3:
    st.metric(
        "음주량",
        f"{input_alcohol}"
    )

st.write("")

# ---------------- 분석 버튼 ----------------
if st.button("🔍 AI 군집 분석 시작"):

    # ---------------- 로딩 ----------------
    with st.spinner("AI 모델 분석 중..."):
        time.sleep(2)

    # ---------------- 사용자 데이터 생성 ----------------
    new_patient = pd.DataFrame(
        [[input_age, input_smoking, input_alcohol]],
        columns=['나이', '흡연', '음주량']
    )

    # ---------------- 스케일링 ----------------
    new_patient_scaled = scaler.transform(new_patient)

    # ---------------- 군집 예측 ----------------
    pred_cluster = model.predict(new_patient_scaled)

    # ---------------- 결과 정보 ----------------
    cluster_info = {
        0: ("🟢 건강군", "낮은 위험 패턴"),
        1: ("🟡 주의군", "생활습관 관리 필요"),
        2: ("🟠 위험군", "정기 검진 권장"),
        3: ("🔴 고위험군", "전문의 상담 권장")
    }

    cluster_name, cluster_desc = cluster_info.get(
        pred_cluster[0],
        ("분석 불가", "정보 없음")
    )

    # ---------------- 결과 출력 ----------------
    st.success(
        f"이 환자는 {pred_cluster[0]}번 군집에 속합니다."
    )

    st.write("")

    result1, result2 = st.columns(2)

    with result1:
        st.metric(
            "군집 결과",
            cluster_name
        )

    with result2:
        st.metric(
            "AI 분석",
            cluster_desc
        )

    # ---------------- 위험도 계산 ----------------
    risk_score = int(
        (input_smoking * 1.2) +
        (input_alcohol * 2) +
        (input_age * 0.3)
    )

    risk_score = min(risk_score, 100)

    st.write("")
    st.subheader("📊 위험도 분석")

    st.progress(risk_score / 100)

    st.caption(f"예상 위험도 점수: {risk_score}/100")

    # ---------------- 군집 그래프 ----------------
    st.write("")
    st.subheader("🧬 군집 그래프 분석")

    # 기존 데이터 스케일링
    X_original = df_original[['나이', '흡연', '음주량']]
    X_scaled_original = scaler.transform(X_original)

    # 기존 데이터 군집 예측
    df_original['cluster'] = model.predict(X_scaled_original)

    # 색상 지정
    colors = {
        0: '#06b6d4',
        1: '#3b82f6',
        2: '#10b981',
        3: '#ef4444'
    }

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    # 다크모드 배경
    fig.patch.set_facecolor('#111827')
    ax.set_facecolor('#111827')

    # 군집별 산점도
    for cluster in sorted(df_original['cluster'].unique()):

        cluster_data = df_original[
            df_original['cluster'] == cluster
        ]

        ax.scatter(
            cluster_data['흡연'],
            cluster_data['음주량'],
            c=colors.get(cluster, 'white'),
            label=f'Cluster {cluster}',
            alpha=0.6,
            s=70
        )

    # 현재 사용자 표시
    ax.scatter(
        input_smoking,
        input_alcohol,
        c='white',
        s=300,
        marker='X',
        edgecolors='black',
        linewidths=2,
        label='현재 사용자'
    )

    # 그래프 스타일
    ax.set_title(
        "AI 군집 내 사용자 위치",
        color='white',
        fontsize=16,
        pad=20
    )

    ax.set_xlabel(
        "흡연 지표",
        color='white'
    )

    ax.set_ylabel(
        "음주량",
        color='white'
    )

    ax.tick_params(colors='white')

    # 축 색상
    for spine in ax.spines.values():
        spine.set_color('#374151')

    # 그리드
    ax.grid(
        True,
        linestyle='--',
        alpha=0.2
    )

    # 범례
    legend = ax.legend()

    for text in legend.get_texts():
        text.set_color("white")

    legend.get_frame().set_facecolor('#1f2937')
    legend.get_frame().set_edgecolor('#374151')

    # Streamlit 출력
    st.pyplot(fig)

    # ---------------- 군집 중심값 ----------------
    st.write("")
    st.subheader("📌 군집 중심값")

    centers = pd.DataFrame(
        scaler.inverse_transform(model.cluster_centers_),
        columns=['나이', '흡연', '음주량'],
        index=[f'Cluster {i}' for i in range(model.n_clusters)]
    )

    st.dataframe(
        centers,
        use_container_width=True
    )

st.markdown('</div>', unsafe_allow_html=True)

#py -m streamlit run lung_cancer.py
