from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from datetime import datetime
import streamlit as st
import numpy as np
import math
import time
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# 기본 설정
# -------------------------------
st.set_page_config(page_title="오늘의 공부 집중 예측기", page_icon="🎯")

data_path = "focus_data.csv"
model_path = "focus_model.pkl"
metrics_path = "metrics.csv"  # train_model.py가 MAE, R² 저장할 파일

# ① 데이터 상태 표시
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
    data_count = len(df)
    st.sidebar.success(f"🗂 현재 누적 설문 수: {data_count}개")
    if "real_focus_time" in df.columns:
        fig, ax = plt.subplots()
        ax.hist(df["real_focus_time"].dropna(), bins=10, color="#4CAF50", alpha=0.7)
        ax.set_xlabel("실제 집중시간(분)")
        ax.set_ylabel("빈도")
        ax.set_title("📊 실제 집중시간 분포")
        st.sidebar.pyplot(fig)
else:
    st.sidebar.warning("❌ focus_data.csv 파일이 없습니다.")

# ② 모델 상태 표시
if os.path.exists(model_path):
    model_info = joblib.load(model_path)
    st.sidebar.success("✅ 학습된 모델이 있습니다!")
    st.sidebar.write(f"📦 특성 수: {len(model_info['columns'])}")
else:
    st.sidebar.warning("⚠️ 아직 학습된 모델이 없습니다.")

# ③ 성능 추세 그래프 (metrics.csv가 있을 때만)
if os.path.exists(metrics_path):
    mdf = pd.read_csv(metrics_path)
    if len(mdf) > 1:
        fig2, ax2 = plt.subplots()
        ax2.plot(mdf["run"], mdf["MAE"], marker="o", label="MAE (오차, ↓좋음)")
        ax2.plot(mdf["run"], mdf["R2"], marker="s", label="R² (정확도, ↑좋음)")
        ax2.set_xlabel("학습 실행 횟수")
        ax2.set_ylabel("값")
        ax2.set_title("📈 모델 성능 추이")
        ax2.legend()
        st.sidebar.pyplot(fig2)


DATA_PATH = "focus_data.csv"        # 전체 사용자 공용 데이터 저장 파일
MODEL_PATH = "focus_model.pkl"      # 학습된 모델 파일 경로

NUMERIC_FEATURES = [
    "sleep_hours", "daytime_sleepiness", "stress_level",
    "caffeine", "last_caf_hour", "exercise_min", "screen_time"
]
CATEGORICAL_FEATURES = {
    "sleep_quality": ["매우 나쁨", "보통", "좋음", "매우 좋음"],
    "mood": ["매우 나쁨", "보통", "좋음", "매우 좋음"],
    "noise": ["매우 조용", "보통", "시끄러움"],
    "place": ["도서관", "스터디카페", "집", "카페"],
    "lighting": ["어두움", "적당함", "밝음"],
    "temperature": ["너무 추움", "적당함", "너무 더움"]
}

# -------------------------------
# 세션 초기화
# -------------------------------
if "page" not in st.session_state:
    st.session_state["page"] = "info"
if "ans" not in st.session_state:
    st.session_state["ans"] = {}

# -------------------------------
# 페이지 이동 함수
# -------------------------------
def go(next_page):
    st.session_state["page"] = next_page
    time.sleep(0.05)
    st.rerun()

# -------------------------------
# Sigmoid 조정 함수
# -------------------------------
def sigmoid_adjust(x):
    return 100 / (1 + math.exp(-0.08 * (x - 50)))

# -------------------------------
# 스코어 계산 블록
# -------------------------------
def sleep_block(ans):
    score = 0
    if 7 <= ans["sleep_hours"] <= 9:
        score += 12
    elif 6 <= ans["sleep_hours"] < 7 or 9 < ans["sleep_hours"] <= 10:
        score += 6
    else:
        score -= 10
    score += {"매우 나쁨": -6, "보통": 0, "좋음": +4, "매우 좋음": +8}[ans["sleep_quality"]]
    if ans["daytime_sleepiness"] >= 15:
        score -= 6
    return score

def stress_block(ans):
    s = ans["stress_level"]
    m = {"매우 나쁨": -5, "보통": 0, "좋음": 3, "매우 좋음": 5}[ans["mood"]]
    score = -0.8 * s + m
    if s >= 7 and m <= 0:
        score -= 3
    return score

def habit_block(ans):
    score = 0
    if 1 <= ans["caffeine"] <= 3:
        score += 4
    elif ans["caffeine"] > 5:
        score -= 3
    if 3 <= ans["last_caf_hour"] <= 8:
        score += 2
    elif ans["last_caf_hour"] <= 2:
        score -= 2
    if ans["breakfast"] == "먹음":
        score += 3
    else:
        score -= 2
    if 15 <= ans["exercise_min"] <= 45:
        score += 3
    elif ans["exercise_min"] > 90 or ans["exercise_min"] < 10:
        score -= 2
    if ans["screen_time"] > 180:
        score -= 6
    elif ans["screen_time"] > 60:
        score -= 3
    return score

def env_block(ans):
    noise_map = {"매우 조용": +4, "보통": +1, "시끄러움": -5}
    place_map = {"도서관": +4, "스터디카페": +3, "집": +1, "카페": -1}
    lighting_adj = {"어두움": -1, "적당함": +1, "밝음": +2}[ans["lighting"]]
    temp_adj = {"너무 추움": -1, "적당함": +1, "너무 더움": -1}[ans["temperature"]]
    return noise_map[ans["noise"]] + place_map[ans["place"]] + lighting_adj + temp_adj

def total_score(ans):
    sleep = sleep_block(ans)
    stress = stress_block(ans)
    habit = habit_block(ans)
    env = env_block(ans)
    raw_score = (0.4 * sleep + 0.3 * stress + 0.2 * habit + 0.1 * env)
    adjusted_score = sigmoid_adjust(raw_score)
    return adjusted_score

# -------------------------------
# 데이터 변환/저장 유틸 함수
# -------------------------------
def _row_from_answers(ans, predicted_score, real_focus_time=None, self_rating=None):
    base = ans.copy()
    base["predicted_score"] = predicted_score
    base["real_focus_time"] = real_focus_time
    base["self_rating"] = self_rating
    base["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return base

def _features_dataframe_from_answers(ans, model_columns):
    row = {k: ans[k] for k in NUMERIC_FEATURES}
    df = pd.DataFrame([row])
    cat = {}
    for k, cats in CATEGORICAL_FEATURES.items():
        for c in cats:
            cat[f"{k}__{c}"] = 1 if ans[k] == c else 0
    cat_df = pd.DataFrame([cat])
    X = pd.concat([df, cat_df], axis=1)
    if model_columns is not None:
        X = X.reindex(columns=model_columns, fill_value=0)
    return X

def _try_predict_with_model(ans):
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    cols = bundle["columns"]
    X = _features_dataframe_from_answers(ans, cols)
    try:
        y_pred = model.predict(X)[0]
        return y_pred
    except:
        return None
def upload_to_drive(local_path, drive_folder_id):
    creds = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"],
        scopes=["https://www.googleapis.com/auth/drive.file"]
    )
    service = build("drive", "v3", credentials=creds)

    file_metadata = {"name": "focus_data.csv", "parents": [drive_folder_id]}
    media = MediaFileUpload(local_path, mimetype="text/csv", resumable=True)

    results = service.files().list(
        q=f"name='focus_data.csv' and '{drive_folder_id}' in parents",
        fields="files(id)"
    ).execute()
    items = results.get("files", [])

    if items:
        file_id = items[0]["id"]
        service.files().update(fileId=file_id, media_body=media).execute()
    else:
        service.files().create(body=file_metadata, media_body=media, fields="id").execute()

    st.sidebar.success("✅ Google Drive에 focus_data.csv 업로드 완료!")

# -------------------------------
# 페이지 ① 사용자 정보 입력
# -------------------------------
if st.session_state["page"] == "info":
    st.title("오늘의 공부 집중 예측기 🎯")
    name = st.text_input("이름:")
    age = st.number_input("나이:", 10, 100, 20)
    agree = st.checkbox("연구 목적으로 데이터 활용에 동의합니다.")
    if st.button("설문 시작", disabled=(not agree or name.strip() == "")):
        st.session_state["name"] = name.strip()
        st.session_state["age"] = int(age)
        go("basic")

# -------------------------------
# 페이지 ② 기본 컨디션
# -------------------------------
elif st.session_state["page"] == "basic":
    st.header("① 기본 컨디션 (5문항)")
    sleep_hours = st.slider("1) 수면시간 (시간)", 0.0, 12.0, 7.0, 0.5)
    sleep_quality = st.select_slider("2) 수면의 질", ["매우 나쁨", "보통", "좋음", "매우 좋음"])
    daytime_sleepiness = st.slider("3) 주간 졸림 (0~24)", 0, 24, 8)
    stress_level = st.slider("4) 스트레스 (0~10)", 0, 10, 4)
    mood = st.select_slider("5) 현재 기분", ["매우 나쁨", "보통", "좋음", "매우 좋음"])
    if st.button("다음 →"):
        st.session_state["ans"].update({
            "sleep_hours": sleep_hours, "sleep_quality": sleep_quality,
            "daytime_sleepiness": daytime_sleepiness, "stress_level": stress_level, "mood": mood
        })
        go("lifestyle")

# -------------------------------
# 페이지 ③ 생활 습관
# -------------------------------
elif st.session_state["page"] == "lifestyle":
    st.header("② 생활 습관 (5문항)")
    caffeine = st.slider("6) 카페인 섭취(잔)", 0, 10, 2)
    last_caf_hour = st.slider("7) 마지막 카페인 섭취 후 경과시간(시간)", 0, 24, 6)
    breakfast = st.radio("8) 아침식사 여부", ["먹음", "안 먹음"])
    exercise_min = st.slider("9) 운동 시간(분)", 0, 120, 20)
    screen_time = st.slider("10) 취침 전 화면시간(분)", 0, 300, 60)
    cols = st.columns(2)
    if cols[0].button("← 이전"): go("basic")
    if cols[1].button("다음 →"):
        st.session_state["ans"].update({
            "caffeine": caffeine, "last_caf_hour": last_caf_hour,
            "breakfast": breakfast, "exercise_min": exercise_min, "screen_time": screen_time
        })
        go("environment")

# -------------------------------
# 페이지 ④ 학습 환경
# -------------------------------
elif st.session_state["page"] == "environment":
    st.header("③ 학습 환경 (4문항)")
    noise = st.selectbox("11) 소음", ["매우 조용", "보통", "시끄러움"])
    place = st.selectbox("12) 장소", ["도서관", "스터디카페", "집", "카페"])
    lighting = st.selectbox("13) 조명 밝기", ["어두움", "적당함", "밝음"])
    temperature = st.select_slider("14) 온도 만족도", ["너무 추움", "적당함", "너무 더움"])
    cols = st.columns(2)
    if cols[0].button("← 이전"): go("lifestyle")
    if cols[1].button("결과 보기"):
        st.session_state["ans"].update({
            "noise": noise, "place": place, "lighting": lighting, "temperature": temperature
        })
        go("result")

# -------------------------------
# 페이지 ⑤ 결과
# -------------------------------
elif st.session_state["page"] == "result":
    st.header("📊 오늘의 집중 예측 결과")
    ans = st.session_state["ans"]
    score = total_score(ans)
    ai_pred_minutes = _try_predict_with_model(ans)
    if score >= 70:
        level = "높음"; mean_time, margin = 420, 60
    elif score >= 40:
        level = "보통"; mean_time, margin = 285, 45
    else:
        level = "낮음"; mean_time, margin = 195, 45
    mean_hr, mean_min = divmod(mean_time, 60)
    st.success(f"🎯 집중 레벨: **{level}**")
    st.metric("예상 집중 가능 시간(기초모델)", f"{mean_time}분 ±{margin}분")
    if ai_pred_minutes is not None:
        st.metric("AI 학습기 예측 시간", f"{ai_pred_minutes:.1f}분")
    else:
        st.info("아직 학습된 모델이 없습니다. 피드백 데이터가 쌓이면 학습 가능!")
    if st.button("👉 피드백 보기"): go("feedback")

# -------------------------------
# 페이지 ⑥ 피드백 + 학습
# -------------------------------
elif st.session_state["page"] == "feedback":
    st.header("📥 오늘의 실제 결과 기록")
    real_focus_time = st.number_input("오늘 실제 순공시간(분)", 0, 600, 0, step=10)
    self_rating = st.selectbox("오늘 실제 집중은 어땠나요?", ["좋음", "보통", "나쁨"])
    if st.button("저장"):
        score = total_score(st.session_state["ans"])
        new_row = _row_from_answers(st.session_state["ans"], score, real_focus_time, self_rating)
        df = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else pd.DataFrame()
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_csv(DATA_PATH, index=False, encoding="utf-8-sig")
        st.success("✅ 오늘의 데이터가 저장되었습니다!")

        upload_to_drive("focus_data.csv", "1z5CNnbVFkkpXxskgnNyvTiQ2hUTHZTKH")

    st.divider()
    if st.button("🔁 모델 재학습"):
        if not os.path.exists(DATA_PATH):
            st.warning("데이터가 아직 없습니다."); st.stop()
        df = pd.read_csv(DATA_PATH)
        df_train = df.dropna(subset=["real_focus_time"]).copy()
        if len(df_train) < 10:
            st.warning("학습에 필요한 데이터가 부족합니다 (10개 이상 필요)."); st.stop()

        X = df_train[NUMERIC_FEATURES].copy()
        for k, cats in CATEGORICAL_FEATURES.items():
            for c in cats:
                X[f"{k}__{c}"] = (df_train[k] == c).astype(int)
        y = df_train["real_focus_time"].astype(float)

        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, r2_score
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)
        st.success(f"🎉 모델 재학습 완료! MAE={mae:.1f}분, R²={r2:.3f}")
    
    if st.button("🏠 처음으로"):
        st.session_state.clear()
        go("info")

