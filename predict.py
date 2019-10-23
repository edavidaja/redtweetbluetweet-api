import joblib

pipe = joblib.load("model/model.joblib")

out = pipe.predict_proba(
    ["families belong together", "government takeover of healthcare"]
)

print(out)
