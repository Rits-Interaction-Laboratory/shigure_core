import mysql.connector

from shigure_core.db.config import config


class EventRepository:

    @staticmethod
    def insert_people(people_id: str, icon_path: str, icon_size: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        cur.execute("INSERT INTO people(id, icon_path, icon_size) VALUES (%s, %s, %s)",
                    (people_id, icon_path, icon_size))
        ctx.close()

    @staticmethod
    def insert_object(object_id: str, icon_path: str, icon_size: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        cur.execute("INSERT INTO object(id, icon_path, icon_size) VALUES (%s, %s, %s)",
                    (object_id, icon_path, icon_size))
        ctx.close()

    @staticmethod
    def insert_camera(camera_name: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = """INSERT INTO camera(camera_name) VALUES (%s)"""
        cur.execute(sql, (camera_name,))
        ctx.close()

    @staticmethod
    def insert_event(event_id: int, people_id: str, object_id: str, camera_id: int, action: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        cur.execute("INSERT INTO event(id, people_id, object_id, camera_id, action) VALUES (%s, %s, %s, %s, %s)",
                    (event_id, people_id, object_id, camera_id, action))
        ctx.close()

    @staticmethod
    def insert_frame(event_id: int, frame_count: int, color_path: str, depth_path: str, point_path: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        cur.execute("INSERT INTO frame(event_id, frame_count, color_path, depth_path, point_path) VALUES (%s, %s, %s, %s, %s)",
                    (event_id, frame_count, color_path, depth_path, point_path))
        ctx.close()

    @staticmethod
    def insert_event_meta(event_id: int, width: int, height: int, x: int, y: int, z: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        cur.execute("INSERT INTO event_metadata(event_id, width, height, x, y, z) VALUES (%s, %s, %s, %s, %s, %s)",
                    (event_id, width, height, x, y, z))
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
