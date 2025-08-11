# MLOps Project: Hate Speech Detection API

This project demonstrates a complete MLOps workflow for building, improving, and containerizing a machine learning model. The goal is to detect hate speech in text and serve the model via a robust FastAPI.

## Key Achievements
* **Data Analysis**: Performed exploratory data analysis on an imbalanced dataset, identifying the need for resampling.
* **Model Improvement**: Successfully addressed class imbalance using **SMOTE** to significantly improve hate speech detection recall from **17% to 54%**.
* **API Development**: Created a RESTful API with **FastAPI** to serve the model predictions.
* **Containerization**: Packaged the entire application in a **Docker container** for portability and deployment.

## Technologies
* **Python**
* **Scikit-learn**
* **FastAPI**
* **Docker**

---

## How to Run the Project 🏃‍♂️

You can run this project locally with Docker.

1.  **Clone the repository**:
    ```bash
    git clone <your_github_repository_url>
    cd <your_project_name>
    ```

2.  **Build the Docker image**:
    ```bash
    docker build -t hate-speech-api .
    ```

3.  **Run the Docker container**:
    ```bash
    docker run -d -p 8000:8000 --name hate_speech_detector hate-speech-api
    ```

4.  **Access the API**:
    Your API is now running at `http://localhost:8000`. You can test it by going to the interactive docs: `http://localhost:8000/docs`.

---

## Model Performance 📊

The final, optimized model provides a strong balance of overall accuracy and high recall on the critical hate speech class.

* **Final Accuracy**: 85.27%
* **Hate Speech Recall (Class 0)**: **54%**
* **Hate Speech Precision (Class 0)**: 29%
* **Hate Speech F1-Score (Class 0)**: 0.38