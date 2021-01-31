import mysql.connector

from shigure_core.db.config import config


class EventRepository:

    @staticmethod
    def insert(camera_id: int, people_id: str, object_id: str, action: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()

        cur.execute("INSERT INTO event(camera_id, people_id, object_id, people_id, action) VALUES (%s, %s, %s, %s, %s)",
                    (camera_id, people_id, object_id, action))

        ctx.close()

    @staticmethod
    def select_with_count(page: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()

        rows = cur.fetchall()

        ctx.close()

        if page == 1:
            return rows[-page * 4:]

        return rows[-page * 4:-(page - 1) * 4]
