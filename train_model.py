import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

DATA_PATH = "focus_data.csv"
MODEL_PATH = "focus_model.pkl"

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

if not os.path.exists(DATA_PATH):
    print("❌ focus_data.csv 파일이 없습니다. 앱에서 설문 데이터를 먼저 저장하세요.")
    exit()

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["real_focus_time"]).copy()

if len(df) < 10:
    print(f"⚠️ 데이터가 부족합니다 ({len(df)}개). 최소 10개 이상 필요합니다.")
    exit()

# X 구성
X = df[NUMERIC_FEATURES].copy()
for k, cats in CATEGORICAL_FEATURES.items():
    for c in cats:
        X[f"{k}__{c}"] = (df[k] == c).astype(int)

y = df["real_focus_time"].astype(float)

# 학습
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

# 성능 기록 저장 (metrics.csv)
metrics_path = "metrics.csv"
record = pd.DataFrame({"run": [len(pd.read_csv(metrics_path)) + 1 if os.path.exists(metrics_path) else 1],
                       "MAE": [mae], "R2": [r2]})
record.to_csv(metrics_path, mode="a", header=not os.path.exists(metrics_path), index=False)

# 모델 저장
joblib.dump({"model": model, "columns": list(X.columns)}, MODEL_PATH)
print(f"✅ 모델 학습 완료!")
print(f"📈 MAE={mae:.1f}분, R²={r2:.3f}")
print(f"💾 저장 완료: {MODEL_PATH}")


# 🔍 중요도(Feature Importance) 저장 (AI 피드백용)
# importances = model.feature_importances_
# cols = list(X.columns)
# imp_df = pd.DataFrame({"Feature": cols, "Importance": importances})
# imp_df = imp_df.sort_values(by="Importance", ascending=False)
# imp_df.to_csv("feature_importance.csv", index=False)
# print("💡 주요 영향 요인 상위 5개:")
# print(imp_df.head(5))
# print("💾 feature_importance.csv 저장 완료")