import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash


class Database:

    def __init__(self):
        self.database_url = os.environ.get("DATABASE_URL")

    # -------------------------
    # Connect to PostgreSQL
    # -------------------------
    def connect(self):
        try:
            return psycopg2.connect(self.database_url)
        except Exception as e:
            print("Database connection error:", e)
            return None

    # -------------------------
    # Initialize Database
    # -------------------------
    def init_db(self):

        try:
            connection = self.connect()
            if not connection:
                return

            cursor = connection.cursor()

            # USERS TABLE
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(80) UNIQUE NOT NULL,
                    email VARCHAR(120) UNIQUE NOT NULL,
                    password VARCHAR(255) NOT NULL,
                    role VARCHAR(20) DEFAULT 'user',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # DISEASE TABLE
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS diseases (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(120) UNIQUE NOT NULL,
                    description TEXT,
                    severity FLOAT NULL,
                    avg_duration_days INT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # PREDICTION HISTORY
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id SERIAL PRIMARY KEY,
                    report_id VARCHAR(64) NOT NULL,
                    user_id INT NOT NULL,
                    patient_name VARCHAR(120),
                    predicted_disease VARCHAR(120),
                    confidence FLOAT,
                    symptoms TEXT,
                    recommended_tests TEXT,
                    prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            # CONTACT TABLE
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contact_messages (
                    id SERIAL PRIMARY KEY,
                    name VARCHAR(120) NOT NULL,
                    email VARCHAR(120) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            connection.commit()
            cursor.close()
            connection.close()

            print("Database initialized successfully")

        except Exception as e:
            print("Database init error:", e)

    # -------------------------
    # Contact Form Save
    # -------------------------
    def save_contact_message(self, name, email, message):

        try:
            connection = self.connect()
            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO contact_messages (name,email,message)
                VALUES (%s,%s,%s)
                """,
                (name, email, message),
            )

            connection.commit()
            cursor.close()
            connection.close()

            return True

        except Exception as e:
            print("Contact save error:", e)
            return False

    # -------------------------
    # Register User
    # -------------------------
    def register_user(self, username, email, password, role="user"):

        try:
            connection = self.connect()
            cursor = connection.cursor()

            hashed_password = generate_password_hash(password)

            cursor.execute(
                """
                INSERT INTO users (username,email,password,role)
                VALUES (%s,%s,%s,%s)
                """,
                (username, email, hashed_password, role),
            )

            connection.commit()
            cursor.close()
            connection.close()

            return {"success": True, "message": "User registered successfully"}

        except Exception as e:

            if "duplicate key" in str(e).lower():
                return {"success": False, "message": "Username or email already exists"}

            return {"success": False, "message": str(e)}

    # -------------------------
    # Login User
    # -------------------------
    def login_user(self, username, password):

        try:
            connection = self.connect()
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                "SELECT id,username,email,password,role FROM users WHERE username=%s",
                (username,),
            )

            user = cursor.fetchone()

            cursor.close()
            connection.close()

            if user and check_password_hash(user["password"], password):

                return {
                    "success": True,
                    "user": user
                }

            return {"success": False, "message": "Invalid username or password"}

        except Exception as e:
            return {"success": False, "message": str(e)}

    # -------------------------
    # Get User by ID
    # -------------------------
    def get_user_by_id(self, user_id):

        try:
            connection = self.connect()
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute(
                "SELECT id,username,email,role FROM users WHERE id=%s",
                (user_id,),
            )

            user = cursor.fetchone()

            cursor.close()
            connection.close()

            return user

        except Exception as e:
            print("Fetch user error:", e)
            return None

    # -------------------------
    # Get Diseases
    # -------------------------
    def get_all_diseases(self):

        try:
            connection = self.connect()
            cursor = connection.cursor(cursor_factory=RealDictCursor)

            cursor.execute("SELECT * FROM diseases ORDER BY created_at DESC")

            results = cursor.fetchall()

            cursor.close()
            connection.close()

            return results

        except Exception as e:
            print("Disease fetch error:", e)
            return []

    # -------------------------
    # Save Prediction
    # -------------------------
    def save_prediction(
        self,
        user_id,
        report_id,
        patient_name,
        predicted_disease,
        confidence,
        symptoms,
        recommended_tests,
    ):

        try:
            connection = self.connect()
            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO prediction_history
                (report_id,user_id,patient_name,predicted_disease,confidence,symptoms,recommended_tests)
                VALUES (%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    report_id,
                    user_id,
                    patient_name,
                    predicted_disease,
                    float(confidence or 0),
                    json.dumps(symptoms or []),
                    ", ".join(recommended_tests or []),
                ),
            )

            connection.commit()
            cursor.close()
            connection.close()

            return True

        except Exception as e:
            print("Prediction save error:", e)
            return False


db = Database()