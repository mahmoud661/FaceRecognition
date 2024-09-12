import psycopg2


def connect_db():
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
    try:
        conn = connect_db()
        cur = conn.cursor()
        create_table_query = '''
        CREATE TABLE IF NOT EXISTS Users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            face_embedding BYTEA
        );
        '''
        cur.execute(create_table_query)
        conn.commit()
        print("Table 'Users' has been created (if it did not exist).")
        cur.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")


def insert_user(name):
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO Users (name) VALUES (%s) RETURNING id;", (name,))
            user_id = cur.fetchone()[0]
            conn.commit()
            cur.close()
            conn.close()
            return user_id
        except Exception as e:
            print(f"Error inserting user: {e}")
            return None


def insert_face_embedding(name, embedding=None):
    conn = connect_db()
    if conn is None:
        return

    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO Users (name, face_embedding) VALUES (%s, %s)",
            (name, embedding)
        )
        conn.commit()
        print(f"Inserted {name}'s face embedding.")
    except Exception as e:
        print(f"Error inserting embedding: {e}")
    finally:
        cur.close()
        conn.close()


def fetch_all_users():
    conn = connect_db()
    if conn:
        try:
            cur = conn.cursor()
            cur.execute("SELECT id, name FROM Users;")
            users = cur.fetchall()
            cur.close()
            conn.close()
            return {user[0]: user[1] for user in users}
        except Exception as e:
            print(f"Error fetching users: {e}")
            return {}


def user_exists(name):
    conn = connect_db()
    if conn is None:
        return False

    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM Users WHERE name = %s", (name,))
        result = cur.fetchone()[0]
        return result > 0
    except Exception as e:
        print(f"Error checking if user exists: {e}")
        return False
    finally:
        cur.close()
        conn.close()


