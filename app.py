from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load model + scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

labels = ["Poor", "Average", "Good"]


# 🏠 Home
@app.route('/')
def home():
    return render_template("index.html")


# 🔹 SINGLE PREDICTION
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input
        data = [
            float(request.form['hours']),
            float(request.form['attendance']),
            float(request.form['previous']),
            float(request.form['sleep']),
            float(request.form['assignments']),
            float(request.form['extra']),
            float(request.form['internet']),
            float(request.form['parent']),
            float(request.form['env']),
        ]

        # 🔥 Feature Engineering (same as training)
        study_efficiency = data[2] / (data[0] + 1)
        data.append(study_efficiency)

        # 🔥 Scaling
        data_scaled = scaler.transform([data])

        # 🔥 Prediction
        prediction = model.predict(data_scaled)[0]
        proba = model.predict_proba(data_scaled)[0]

        result = labels[prediction]
        confidence = round(max(proba) * 100, 2)

        return render_template(
            "index.html",
            result=result,
            confidence=confidence
        )

    except Exception as e:
        print(e)
        return render_template("index.html", result="Error")


# 🔹 BULK PREDICTION (CSV)
@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['POST'])
def upload():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # 🔥 AUTO ADD MISSING COLUMNS (IMPORTANT FIX)
        default_cols = {
            "extra_classes": 0,
            "internet": 1,
            "parent_edu": 2,
            "study_env": 2
        }

        for col, val in default_cols.items():
            if col not in df.columns:
                df[col] = val

        # 🔥 Feature Engineering
        df["study_efficiency"] = df["previous"] / (df["hours"] + 1)

        # 🔥 Required Features
        features = [
            'hours', 'attendance', 'previous', 'sleep',
            'assignments', 'extra_classes', 'internet',
            'parent_edu', 'study_env', 'study_efficiency'
        ]

        X = df[features]

        # 🔥 Scaling
        X_scaled = scaler.transform(X)

        # 🔥 Prediction
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        df['Prediction'] = [labels[p] for p in predictions]
        df['Confidence (%)'] = [round(max(p)*100, 2) for p in probabilities]

        return render_template(
            "index.html",
            tables=[df.to_html(classes='table table-striped', index=False)]
        )

    except Exception as e:
        print(e)
        return f"Error: {e}"

        X = df[features]

        # 🔥 Scaling
        X_scaled = scaler.transform(X)

        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)

        df['Prediction'] = [labels[p] for p in predictions]
        df['Confidence (%)'] = [round(max(p)*100, 2) for p in probabilities]

        return render_template(
            "index.html",
            tables=[df.to_html(classes='table table-striped', index=False)]
        )

    except Exception as e:
        print(e)
        return f"Error: {e}"


if __name__ == "__main__":
    app.run(debug=True)