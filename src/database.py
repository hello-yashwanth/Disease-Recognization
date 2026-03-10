import mysql.connector
from mysql.connector import Error
from werkzeug.security import generate_password_hash, check_password_hash
import os

class Database:
    def __init__(self):
        self.host = os.environ.get('DB_HOST', 'localhost')
        self.user = os.environ.get('DB_USER', 'root')
        self.password = os.environ.get('DB_PASSWORD', '')
        self.database = os.environ.get('DB_NAME', 'disease_recognition')
        self.connection = None
    
    def connect(self):
        """Establish connection to MySQL database"""
        try:
            self.connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password,
                database=self.database
            )
            return self.connection
        except Error as e:
            print(f"Error while connecting to MySQL: {e}")
            return None
    
    def disconnect(self):
        """Close database connection"""
        if self.connection and self.connection.is_connected():
            self.connection.close()
    
    def init_db(self):
        """Initialize database tables"""
        try:
            connection = mysql.connector.connect(
                host=self.host,
                user=self.user,
                password=self.password
            )
            cursor = connection.cursor()
            
            # Create database if not exists
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
            connection.database = self.database
            
            # Create users table
            create_users_table = """
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(80) UNIQUE NOT NULL,
                email VARCHAR(120) UNIQUE NOT NULL,
                password VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            cursor.execute(create_users_table)
            connection.commit()
            cursor.close()
            connection.close()
            print("Database initialized successfully")
        except Error as e:
            print(f"Error initializing database: {e}")
    
    def register_user(self, username, email, password):
        """Register a new user"""
        try:
            connection = self.connect()
            if not connection:
                return {'success': False, 'message': 'Database connection failed'}
            
            cursor = connection.cursor()
            hashed_password = generate_password_hash(password)
            
            insert_query = "INSERT INTO users (username, email, password) VALUES (%s, %s, %s)"
            cursor.execute(insert_query, (username, email, hashed_password))
            connection.commit()
            cursor.close()
            connection.close()
            
            return {'success': True, 'message': 'User registered successfully'}
        except Error as e:
            if 'Duplicate entry' in str(e):
                return {'success': False, 'message': 'Username or email already exists'}
            return {'success': False, 'message': f'Registration failed: {str(e)}'}
    
    def login_user(self, username, password):
        """Authenticate user"""
        try:
            connection = self.connect()
            if not connection:
                return {'success': False, 'message': 'Database connection failed'}
            
            cursor = connection.cursor(dictionary=True)
            query = "SELECT id, username, email, password FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            cursor.close()
            connection.close()
            
            if user and check_password_hash(user['password'], password):
                return {
                    'success': True,
                    'message': 'Login successful',
                    'user': {
                        'id': user['id'],
                        'username': user['username'],
                        'email': user['email']
                    }
                }
            else:
                return {'success': False, 'message': 'Invalid username or password'}
        except Error as e:
            return {'success': False, 'message': f'Login failed: {str(e)}'}
    
    def get_user(self, username):
        """Get user by username"""
        try:
            connection = self.connect()
            if not connection:
                return None
            
            cursor = connection.cursor(dictionary=True)
            query = "SELECT id, username, email FROM users WHERE username = %s"
            cursor.execute(query, (username,))
            user = cursor.fetchone()
            cursor.close()
            connection.close()
            
            return user
        except Error as e:
            print(f"Error fetching user: {e}")
            return None
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        try:
            connection = self.connect()
            if not connection:
                return None
            
            cursor = connection.cursor(dictionary=True)
            query = "SELECT id, username, email FROM users WHERE id = %s"
            cursor.execute(query, (user_id,))
            user = cursor.fetchone()
            cursor.close()
            connection.close()
            
            return user
        except Error as e:
            print(f"Error fetching user by ID: {e}")
            return None

# Create a global database instance
db = Database()
