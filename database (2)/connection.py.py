import mysql.connector
from mysql.connector import Error

# Database configuration
DB_CONFIG = {
    'host': 'localhost',          # or your MySQL server IP
    'user': 'root',               # replace with your MySQL username
    'password': 'your_password',  # replace with your MySQL password
    'database': 'AgroSmart'       # make sure schema name matches
}

def get_connection():
    """
    Establish a connection to the MySQL database and return the connection object.
    """
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            print("✅ Connected to MySQL database")
            return conn
    except Error as e:
        print(f"❌ Error connecting to MySQL: {e}")
        return None

def close_connection(conn):
    """
    Close the given database connection.
    """
    try:
        if conn and conn.is_connected():
            conn.close()
            print("🔒 Database connection closed")
    except Error as e:
        print(f"❌ Error closing connection: {e}")

# Utility function to get a cursor safely
def get_cursor(conn):
    """
    Returns a cursor object from the given connection.
    """
    if conn and conn.is_connected():
        return conn.cursor()
    else:
        print("❌ Connection is not established.")
        return None

# Example usage
if __name__ == "__main__":
    conn = get_connection()
    if conn:
        cursor = get_cursor(conn)
        if cursor:
            cursor.execute("SHOW TABLES;")
            tables = cursor.fetchall()
            print("Tables in AgroSmart:", tables)
            cursor.close()
        close_connection(conn)
