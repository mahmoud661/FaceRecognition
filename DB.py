import psycopg2
import numpy as np

def connect_db():
    """Establish a connection to the PostgreSQL database."""
    print("Connecting to the database...")
    try:
        conn = psycopg2.connect(
            host="localhost",
            database="FaceRecognition",
            user="postgres",
            password="123456",
            port=5432
        )
        print("Connected to the database.")
        return conn
    except Exception as e:
        print(f"Error connecting to the database: {e}")
        return None

def create_table_if_not_exists():
    """Create the Users table if it does not already exist."""
    conn = connect_db()
    if conn:
        try:
            with conn.cursor() as cur:
                create_table_query = '''
                CREATE TABLE IF NOT EXISTS Users (
                    id SERIAL PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    face_embedding BYTEA
                );
                CREATE TABLE IF NOT EXISTS UserImages (
                    id SERIAL PRIMARY KEY,
                    user_name TEXT REFERENCES Users(name),
                    image_path TEXT
                );
                '''
                cur.execute(create_table_query)
                conn.commit()
                print("Tables 'Users' and 'UserImages' have been created (if they did not exist).")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            conn.close()

def insert_user(name):
    """Insert a new user into the Users table."""
    conn = connect_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO Users (name) VALUES (%s) RETURNING id;", (name,))
                user_id = cur.fetchone()[0]
                conn.commit()
                print(f"User {name} inserted with ID {user_id}.")
                return user_id
        except Exception as e:
            print(f"Error inserting user: {e}")
            return None
        finally:
            conn.close()

def insert_face_embedding(name, embedding=None):
    """Insert or update a user's face embedding."""
    conn = connect_db()
    if conn:
        try:
            with conn.cursor() as cur:
                if embedding is not None:
                    embedding_bytes = embedding.tobytes()
                    cur.execute(
                        "UPDATE Users SET face_embedding = %s WHERE name = %s;",
                        (psycopg2.Binary(embedding_bytes), name))
                    conn.commit()
                    print(f"Updated face embedding for {name}.")
                else:
                    print("No embedding provided.")
        except Exception as e:
            print(f"Error inserting face embedding: {e}")
        finally:
            conn.close()

def insert_user_image(name, image_path):
    """Insert the path of a user's face image."""
    conn = connect_db()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO UserImages (user_name, image_path) VALUES (%s, %s);",
                    (name, image_path))
                conn.commit()
                print(f"Inserted image path for {name}: {image_path}.")
        except Exception as e:
            print(f"Error inserting image path: {e}")
        finally:
            conn.close()

def fetch_all_user_embeddings():
    """Fetch all user embeddings from the database."""
    conn = connect_db()
    embeddings = {}
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT name, face_embedding FROM Users;")
                rows = cur.fetchall()
                for name, embedding in rows:
                    embeddings[name] = np.frombuffer(embedding, dtype=np.float64)
        except Exception as e:
            print(f"Error fetching embeddings: {e}")
        finally:
            conn.close()
    return embeddings

def user_exists(name):
    """Check if a user exists in the database."""
    conn = connect_db()
    exists = False
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT EXISTS(SELECT 1 FROM Users WHERE name=%s);", (name,))
                exists = cur.fetchone()[0]
        except Exception as e:
            print(f"Error checking user existence: {e}")
        finally:
            conn.close()
    return exists
