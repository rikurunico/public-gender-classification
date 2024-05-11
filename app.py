from flask import Flask, render_template, jsonify, request
from flask_wtf import CSRFProtect
from joblib import load

# Inisialisasi aplikasi Flask
app = Flask(__name__)

# Muat model dan vectorizer
model_naive_bayes = load("model/model_naive_bayes.joblib")
model_knn = load("model/model_knn.joblib")
model_svm = load("model/model_svm.joblib")
model_gradient_boosting = load("model/model_gradient_boosting.joblib")
model_logistic_regression = load("model/model_regresi_logistik.joblib")
model_random_forest = load("model/model_random_forest.joblib")
vectorizer = load("model/vectorizer.joblib")

app.config["SECRET_KEY"] = "kunci_rahasiamu"  # Gunakan kunci rahasia yang kuat

# Inisialisasi CSRF Protection
csrf = CSRFProtect(app)


# Fungsi root untuk route '/'
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# Endpoint untuk prediksi
@app.route("/predict", methods=["POST"])
def predict():
    # Ambil data JSON yang dikirim dari klien
    data = request.get_json()
    # Ambil nama dari data JSON
    name = data["name"]
    # Transformasikan input menggunakan vectorizer
    name_vectorized = vectorizer.transform([name])
    # Lakukan prediksi dengan Semua model

    # Naive Bayes
    result_naive_bayes = model_naive_bayes.predict(name_vectorized)[0]
    accuracies_naive_bayes = model_naive_bayes.predict_proba(name_vectorized)[0]

    # KNN
    result_knn = model_knn.predict(name_vectorized)[0]
    accuracies_knn = model_knn.predict_proba(name_vectorized)[0]

    # SVM
    result_svm = model_svm.predict(name_vectorized)[0]
    accuracies_svm = model_svm.predict_proba(name_vectorized)[0]

    # Gradient Boosting
    result_gradient_boosting = model_gradient_boosting.predict(name_vectorized)[0]
    accuracies_gradient_boosting = model_gradient_boosting.predict_proba(
        name_vectorized
    )[0]

    # Regresi Logistik
    result_logistic_regression = model_logistic_regression.predict(name_vectorized)[0]
    accuracies_logistic_regression = model_logistic_regression.predict_proba(
        name_vectorized
    )[0]

    # Random Forest
    result_random_forest = model_random_forest.predict(name_vectorized)[0]
    accuracies_random_forest = model_random_forest.predict_proba(name_vectorized)[0]

    # Buat respons JSON dengan hasil prediksi
    response = {
        "naive_bayes": {
            "prediction": result_naive_bayes,
            "accuracies": accuracies_naive_bayes.tolist(),
        },
        "knn": {"prediction": result_knn, "accuracies": accuracies_knn.tolist()},
        "svm": {"prediction": result_svm, "accuracies": accuracies_svm.tolist()},
        "gradient_boosting": {
            "prediction": result_gradient_boosting,
            "accuracies": accuracies_gradient_boosting.tolist(),
        },
        "logistic_regression": {
            "prediction": result_logistic_regression,
            "accuracies": accuracies_logistic_regression.tolist(),
        },
        "random_forest": {
            "prediction": result_random_forest,
            "accuracies": accuracies_random_forest.tolist(),
        },
    }

    return jsonify(response)

    # result = model_naive_bayes.predict(name_vectorized)[0]
    # accuracies = model_naive_bayes.predict_proba(name_vectorized)[0]
    # Buat respons JSON dengan hasil prediksi
    # if result == "L":
    #     prediction = (
    #         f"{name} adalah laki-laki dengan probabilitas {accuracies[0] * 100:.2f}%"
    #     )
    # else:
    #     prediction = (
    #         f"{name} adalah perempuan dengan probabilitas {accuracies[1] * 100:.2f}%"
    #     )
    # return jsonify(
    #     {"prediction": prediction, "result": result, "accuracies": accuracies.tolist()}
    # )


# Jalankan aplikasi
if __name__ == "__main__":
    app.run(debug=True)
