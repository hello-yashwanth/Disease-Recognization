import psycopg2
from werkzeug.security import generate_password_hash

DATABASE_URL = "postgresql://diseasedb_inay_user:4ZW1aHgXJyJkTDesQHcKvPPmU5Mi5SHT@dpg-d6pd8skr85hc739j3hjg-a.singapore-postgres.render.com/diseasedb_inay"

# Connect to database
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Generate hashed password
password = generate_password_hash("123456789")

# Insert user
cursor.execute(
    """
    INSERT INTO users (username, email, password, role)
    VALUES (%s, %s, %s, %s)
    """,
    ("yash", "admin@test.com", password, "user")
)

# Save changes
conn.commit()

# Close connection
cursor.close()
conn.close()

print("User created successfully")