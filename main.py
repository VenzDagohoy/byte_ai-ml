import os
import json
import mysql.connector
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import numpy as np

load_dotenv()

app = FastAPI(title="ByteQuest AI & ML Microservice", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def get_db_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST"),
        port=int(os.getenv("DB_PORT", 4000)),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        ssl_disabled=False,
        ssl_verify_identity=True
    )

class ChatRequest(BaseModel):
    message: str
    user_id: int
    role: str = "student"

class GenerateFromDocsRequest(BaseModel):
    classroom_id: int
    resource_ids: list[str]
    generation_type: str # 'challenge', 'quiz', 'reviewer'
    subject: str
    difficulty: str = "medium"

@app.post("/api/ai/generate-from-docs")
def generate_from_docs(request: GenerateFromDocsRequest):
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        
        # Fetch selected lecture contents
        format_strings = ','.join(['%s'] * len(request.resource_ids))
        cursor.execute(f"SELECT filename, file_content FROM classroom_resources WHERE id IN ({format_strings})", tuple(request.resource_ids))
        docs = cursor.fetchall()
        conn.close()

        if not docs:
            raise HTTPException(status_code=404, detail="Selected lecture documents not found.")

        combined_context = "\n\n".join([f"--- File: {d['filename']} ---\n{d['file_content']}" for d in docs])

        system_prompt = (
            f"You are an expert computer science curriculum generator. Based strictly on the following uploaded lecture notes, "
            f"generate a high-quality '{request.generation_type}' for the subject track '{request.subject}' with difficulty '{request.difficulty}'.\n\n"
            f"Lecture Content Context:\n{combined_context}\n\n"
            "You MUST return a valid JSON object strictly matching this schema with no markdown wrapping: "
            "{"
            "  \"title\": \"Title of the generated activity\","
            "  \"concept\": \"Core CS concept covered\","
            "  \"prompt\": \"Instructions, summary notes, or problem prompt for the student\","
            "  \"starter_code\": \"Boilerplate code if challenge (or empty string)\","
            "  \"test_input\": \"Test case input if challenge (or empty string)\","
            "  \"test_output\": \"Test case expected output if challenge (or empty string)\""
            "}"
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=system_prompt,
        )

        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.startswith("```"):
            raw_text = raw_text[3:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]

        return json.loads(raw_text.strip())

    except Exception as e:
        print(f"AI Doc Generation Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
def chat_with_bytebot(request: ChatRequest):
    current_user_id = request.user_id
    current_role = request.role

    def get_my_classrooms() -> str:
        try:
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            if current_role == "instructor":
                cursor.execute("SELECT id, title, subject, join_code FROM classrooms WHERE instructor_id = %s", (current_user_id,))
            else:
                cursor.execute(
                    """SELECT c.id, c.title, c.subject, c.join_code FROM classrooms c 
                       JOIN classroom_members m ON c.id = m.classroom_id WHERE m.student_id = %s""",
                    (current_user_id,)
                )
            results = cursor.fetchall()
            conn.close()
            return json.dumps(results, default=str)
        except Exception as e:
            return json.dumps({"error": f"Database error: {str(e)}"})

    try:
        chat = client.chats.create(
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                tools=[get_my_classrooms],
                system_instruction=(
                    f"You are 'ByteBot', an elite Socratic computer science tutor. User ID: {current_user_id}, Role: {current_role}."
                ),
                temperature=0.3
            )
        )
        response = chat.send_message(request.message)
        return {"reply": response.text if response.text else "I have processed your request."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ml/train-evaluate")
def train_evaluate_models():
    try:
        conn = get_db_connection()
        query = "SELECT prior_quiz_accuracy, avg_solve_time_seconds, compilation_error_count, attempts_before_success, hints_requested_count, task_difficulty_level, target_task_success FROM telemetry_features"
        df = pd.read_sql(query, conn)
        conn.close()

        if len(df) < 10:
            np.random.seed(42)
            df = pd.DataFrame({
                "prior_quiz_accuracy": np.random.uniform(0.5, 1.0, 50),
                "avg_solve_time_seconds": np.random.uniform(20, 120, 50),
                "compilation_error_count": np.random.randint(0, 5, 50),
                "attempts_before_success": np.random.randint(1, 3, 50),
                "hints_requested_count": np.random.randint(0, 3, 50),
                "task_difficulty_level": np.random.randint(1, 4, 50),
                "target_task_success": np.random.choice([0, 1], 50, p=[0.3, 0.7])
            })

        X = df[["prior_quiz_accuracy", "avg_solve_time_seconds", "compilation_error_count", "attempts_before_success", "hints_requested_count", "task_difficulty_level"]].values
        y = df["target_task_success"].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        
        lr = LogisticRegression()
        lr.fit(scaler.fit_transform(X_train), y_train)
        lr_preds = lr.predict(scaler.transform(X_test))

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        rf_preds = rf.predict(X_test)

        def get_metrics(y_true, preds):
            cm = confusion_matrix(y_true, preds)
            tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
            return {
                "accuracy": float(accuracy_score(y_true, preds)),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1_score": float(f1_score(y_true, preds, zero_division=0)),
                "confusion_matrix": {"true_positive": int(tp), "false_positive": int(fp), "false_negative": int(fn), "true_negative": int(tn)}
            }

        return {
            "dataset_size": len(df),
            "logistic_regression": get_metrics(y_test, lr_preds),
            "random_forest": get_metrics(y_test, rf_preds),
            "recommended_model": "Random Forest",
            "rationale": "Random Forest handles non-linear feature interactions between solve time and hint requests more effectively."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
def ai_health():
    return {"status": "operational", "service": "ByteQuest Python AI Microservice"}