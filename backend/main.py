import sqlite3
import os
import json
import re
import requests
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize Flask App
app = Flask(__name__, template_folder="templates")
CORS(app)

# Load API Keys
HF_TOKEN = os.getenv("HF_TOKEN")          # <--- set this in your .env
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

# Ensure 'data' folder exists
if not os.path.exists("data"):
    os.makedirs("data")

# Database file inside 'data' folder
DB_FILE = os.path.join("data", "memory.db")


# Database Initialization
def init_db():
    """Creates the database if it doesn't exist"""
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS research_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task TEXT UNIQUE,
                hypotheses TEXT,
                web_research TEXT,
                analysis TEXT,
                reasoning TEXT,
                evaluation TEXT,
                summary TEXT,
                conclusion TEXT
            )
            """
        )
        conn.commit()


init_db()  # Ensure the database is initialized


# Helper: clean/normalize headings in model text
def clean_headings(text: str) -> str:
    # 1. Remove composite labels you don't want at all
    remove_patterns = [
        r"\*\*\s*Multiple Perspectives\s*:\s*",
        r"\*\s*Multiple Perspectives\s*:\s*",
        r"Multiple Perspectives\s*:\s*",
    ]
    for pat in remove_patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE)

    # 2. Remove all asterisks so no **bold** or *list* markers remain
    text = text.replace("*", "")

    # 3. Normalize common headings to clean form "Heading:"
    normalize_map = {
        "analysis": "Analysis",
        "summary": "Summary",
        "conclusion": "Conclusion",
        "evaluation": "Evaluation",
        "reasoning": "Reasoning",
        "hypotheses": "Hypotheses",
        "key insights": "Key Insights",
        "recommendation": "Recommendation",
    }
    for key, proper in normalize_map.items():
        pattern = rf"(?mi)^\s*{key}\s*:\s*"
        replacement = f"{proper}: "
        text = re.sub(pattern, replacement, text)

    return text.strip()


# Function to Fetch AI Responses via Hugging Face Router (OpenAI-compatible)
def fetch_ai_response(prompt: str) -> str:
    if not HF_TOKEN:
        return "AI Request failed: HF_TOKEN is not set in environment variables"

    client = OpenAI(
        base_url="https://router.huggingface.co/v1",
        api_key=HF_TOKEN,
    )

    system_prompt = (
        "You are a clear, step-by-step research assistant. "
        "Answer in simple English using short paragraphs and numbered/bullet lists when useful. "
        "Avoid custom labels like 'Multiple Perspectives:'. "
        "Headings should be simple plain text like 'Summary:' or 'Recommendation:'. "
        "Do not use Markdown bold or asterisks."
    )

    try:
        completion = client.chat.completions.create(
            model="meta-llama/Llama-3.1-8B-Instruct:novita",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=512,
        )
        raw_text = completion.choices[0].message.content
        cleaned = clean_headings(raw_text)
        return cleaned
    except Exception as e:
        return f"AI Request failed: {str(e)}"


# Web Extractor: Retrieves Information from the Web
def fetch_web_results(query: str):
    search_url = "https://serpapi.com/search"
    params = {
        "q": query,
        "engine": "google_scholar",
        "api_key": SERPAPI_KEY,
        "num": 5,
    }

    try:
        response = requests.get(search_url, params=params)
        response.raise_for_status()
        results = response.json()

        if "organic_results" not in results:
            return []

        formatted_results = [
            {
                "title": item.get("title", "No Title"),
                "link": item.get("link", "No Link"),
                "snippet": item.get("snippet", "No Description Available"),
            }
            for item in results.get("organic_results", [])[:5]
        ]

        return formatted_results

    except requests.exceptions.RequestException:
        return []


# Store data in SQLite
def store_in_db(
    task,
    hypothesis,
    web_research,
    analysis,
    reasoning,
    evaluation,
    summary,
    conclusion,
):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO research_data 
                (task, hypotheses, web_research, analysis, reasoning, evaluation, summary, conclusion)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(task) DO UPDATE SET
                hypotheses=excluded.hypotheses,
                web_research=excluded.web_research,
                analysis=excluded.analysis,
                reasoning=excluded.reasoning,
                evaluation=excluded.evaluation,
                summary=excluded.summary,
                conclusion=excluded.conclusion
            """,
            (
                task,
                hypothesis,
                json.dumps(web_research),
                analysis,
                reasoning,
                evaluation,
                summary,
                conclusion,
            ),
        )
        conn.commit()


# Retrieve stored task from database
def get_from_db(task: str):
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM research_data WHERE task = ?", (task,))
        row = cursor.fetchone()

        if row:
            return {
                "Task": row[1],
                "Generated Hypotheses": row[2].split("\n"),
                "Web Research": json.loads(row[3]),
                "Analysis": row[4].split("\n"),
                "Reasoning": row[5].split("\n"),
                "Evaluation": row[6].split("\n"),
                "Summary": row[7].split("\n"),
                "Conclusion": row[8].split("\n"),
            }
        return None


# Supervisor Agent: Handles All AI & Web Processing
@app.route("/supervisor", methods=["POST"])
def supervisor_agent():
    data = request.json
    task = data.get("task", "").strip()

    if not task:
        return jsonify({"error": "Missing task parameter"}), 400

    # Check if task exists in DB
    stored_result = get_from_db(task)
    if stored_result:
        return jsonify(stored_result)

    # Step 1: Generate Hypotheses
    hypothesis_prompt = f"Generate possible hypotheses based on: {task}"
    hypothesis_response = fetch_ai_response(hypothesis_prompt)

    # Step 2: Fetch Web Results
    web_results = fetch_web_results(task)

    # Step 3: Perform Analysis
    analysis_prompt = (
        "Analyze the following hypotheses and web research:\n"
        f"Hypotheses:\n{hypothesis_response}\n\nWeb Research:\n{web_results}"
    )
    analysis_response = fetch_ai_response(analysis_prompt)

    # Step 4: Provide Logical Reasoning
    reasoning_prompt = f"Provide logical reasoning based on this analysis:\n{analysis_response}"
    reasoning_response = fetch_ai_response(reasoning_prompt)

    # Step 5: Evaluate Different Perspectives
    evaluation_prompt = (
        f"Evaluate different perspectives based on the reasoning:\n{reasoning_response}"
    )
    evaluation_response = fetch_ai_response(evaluation_prompt)

    # Step 6: Summarize Key Insights
    summary_prompt = f"Summarize key insights from evaluation:\n{evaluation_response}"
    summary_response = fetch_ai_response(summary_prompt)

    # Step 7: Provide Conclusion
    conclusion_prompt = (
        f"Based on the summary, provide a structured conclusion:\n{summary_response}"
    )
    conclusion_response = fetch_ai_response(conclusion_prompt)

    # Store in Database
    store_in_db(
        task,
        hypothesis_response,
        web_results,
        analysis_response,
        reasoning_response,
        evaluation_response,
        summary_response,
        conclusion_response,
    )

    # Return the structured response
    formatted_response = {
        "Task": task,
        "Generated Hypotheses": hypothesis_response,
        "Web Research": web_results,
        "Analysis": analysis_response.split("\n"),
        "Reasoning": reasoning_response.split("\n"),
        "Evaluation": evaluation_response.split("\n"),
        "Summary": summary_response.split("\n"),
        "Conclusion": conclusion_response.split("\n"),
    }

    return jsonify(formatted_response)


# Fetch past queries
@app.route("/past_queries", methods=["GET"])
def get_past_queries():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT task FROM research_data")
        rows = cursor.fetchall()

    past_queries = [row[0] for row in rows] if rows else []
    return jsonify({"past_queries": past_queries})


# Homepage Route
@app.route("/")
def home():
    return render_template("index.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
