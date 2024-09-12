CREATE TABLE IF NOT EXISTS Users (
            id SERIAL PRIMARY KEY,
            name TEXT NOT NULL,
            face_embedding BYTEA NOT NULL

);