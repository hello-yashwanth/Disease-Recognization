import os
import json
import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash


class Database:

    def __init__(self):
        self.host = os.environ.get("DB_HOST", "localhost")
        self.user = os.environ.get("DB_USER", "root")
        self.password = os.environ.get("DB_PASSWORD", "root")
        self.database = os.environ.get("DB_NAME", "disease_recognition")

    # -------------------------
    # Connect to MySQL
    # -------------------------
    def connect(self):
        try:
            return mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database,
            )
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return None

    # -------------------------
    # Initialize Database
    # -------------------------
    def init_db(self):

        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
            )

            cursor = connection.cursor()

            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            connection.database = self.database

            # USERS TABLE
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
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
                    id INT AUTO_INCREMENT PRIMARY KEY,
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
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    report_id VARCHAR(64) NOT NULL,
                    user_id INT NOT NULL,
                    patient_name VARCHAR(120),
                    predicted_disease VARCHAR(120),
                    confidence FLOAT,
                    symptoms TEXT,
                    recommended_tests TEXT,
                    prediction_date DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(id)
                )
            """)

            # CONTACT TABLE
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS contact_messages (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    name VARCHAR(120) NOT NULL,
                    email VARCHAR(120) NOT NULL,
                    message TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Migrations
            for alter in (
                "ALTER TABLE prediction_history ADD COLUMN confidence FLOAT",
                "ALTER TABLE prediction_history ADD COLUMN symptoms TEXT",
                "ALTER TABLE diseases ADD COLUMN severity FLOAT NULL",
                "ALTER TABLE diseases ADD COLUMN avg_duration_days INT NULL",
            ):
                try:
                    cursor.execute(alter)
                except Error as e:
                    if getattr(e, 'errno', None) != 1060:
                        print(f"Schema migration error: {e}")

            connection.commit()
            cursor.close()
            connection.close()

            print("Database initialized successfully")

        except Error as e:
            print(f"Error initializing database: {e}")

    # -------------------------
    # Contact Form Save
    # -------------------------
    def save_contact_message(self, name, email, message):

        try:
            connection = self.connect()

            if not connection:
                return False

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO contact_messages (name, email, message)
                VALUES (%s, %s, %s)
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

            if not connection:
                return {"success": False, "message": "Database connection failed"}

            cursor = connection.cursor()

            hashed_password = generate_password_hash(password)

            cursor.execute(
                "INSERT INTO users (username, email, password, role) VALUES (%s,%s,%s,%s)",
                (username, email, hashed_password, role),
            )

            connection.commit()

            cursor.close()
            connection.close()

            return {"success": True, "message": "User registered successfully"}

        except Error as e:

            if "Duplicate entry" in str(e):
                return {"success": False, "message": "Username or email already exists"}

            return {"success": False, "message": f"Registration failed: {str(e)}"}

    # -------------------------
    # Login User
    # -------------------------
    def login_user(self, username, password):

        try:
            connection = self.connect()

            if not connection:
                return {"success": False, "message": "Database connection failed"}

            cursor = connection.cursor(dictionary=True)

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
                    "message": "Login successful",
                    "user": {
                        "id": user["id"],
                        "username": user["username"],
                        "email": user["email"],
                        "role": user.get("role", "user"),
                    },
                }

            return {"success": False, "message": "Invalid username or password"}

        except Error as e:
            return {"success": False, "message": f"Login failed: {str(e)}"}

    # -------------------------
    # Get User by ID
    # -------------------------
    def get_user_by_id(self, user_id):

        try:
            connection = self.connect()

            if not connection:
                return None

            cursor = connection.cursor(dictionary=True)

            cursor.execute(
                "SELECT id,username,email,role FROM users WHERE id=%s",
                (user_id,),
            )

            user = cursor.fetchone()

            cursor.close()
            connection.close()

            return user

        except Error as e:
            print(f"Error fetching user by ID: {e}")
            return None

    # -------------------------
    # Prediction Stats
    # -------------------------
    def get_prediction_stats(self):

        connection = self.connect()

        cursor = connection.cursor(dictionary=True)

        cursor.execute("""
            SELECT predicted_disease, COUNT(*) as count
            FROM prediction_history
            GROUP BY predicted_disease
            ORDER BY count DESC
            LIMIT 5
        """)

        results = cursor.fetchall()

        cursor.close()
        connection.close()

        return results

    # -------------------------
    # Diseases
    # -------------------------
    def add_disease(self, name, description, severity=None, avg_duration_days=None):

        try:
            connection = self.connect()

            if not connection:
                return False

            cursor = connection.cursor()

            cursor.execute(
                "INSERT INTO diseases (name,description,severity,avg_duration_days) VALUES (%s,%s,%s,%s)",
                (name, description, severity, avg_duration_days),
            )

            connection.commit()

            cursor.close()
            connection.close()

            return True

        except Exception as e:
            print(f"Error adding disease: {e}")
            return False

    def get_all_diseases(self):

        try:
            connection = self.connect()

            if not connection:
                return []

            cursor = connection.cursor(dictionary=True)

            cursor.execute("SELECT * FROM diseases ORDER BY created_at DESC")

            results = cursor.fetchall()

            cursor.close()
            connection.close()

            return results

        except Exception as e:
            print(f"Error fetching diseases: {e}")
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

            if not connection:
                return False

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
                    float(confidence or 0.0),
                    json.dumps(symptoms or []),
                    ", ".join(recommended_tests or []),
                ),
            )

            connection.commit()

            cursor.close()
            connection.close()

            return True

        except Exception as e:
            print(f"Error saving prediction: {e}")
            return False


db = Database()