import mysql.connector

from shigure_core.db.config import config
from shigure_core.db.convert_format import ConvertMsg


class EventRepository:

    @staticmethod
    def insert_people(person_id: str, icon_path: str, icon_width: int, icon_height: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO people(person_id, icon_path, icon_width, icon_height) VALUES ('{person_id}', '{icon_path}', '{icon_width}', '{icon_height}')"
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_object(object_id: str, icon_path: str, icon_width: int, icon_height: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO object(object_id, icon_path, icon_width, icon_height) VALUES ('{object_id}', '{icon_path}', '{icon_width}', '{icon_height}')"
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_camera(name: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO camera(name) VALUES ('{name}')"
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_event(event_id: str, people_id: int, object_id: int, camera_id: int, action: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO event(id, people_id, object_id, camera_id, action) VALUES ('{event_id}', {people_id}, '{object_id}', {camera_id}, '{action}') "
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_frame(event_id: str, frame_count: int, color_path: str, depth_path: str, points_path: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO frame(event_id, frame_count, color_path, depth_path, points_path) VALUES ('{event_id}', {frame_count}, '{color_path}', '{depth_path}', '{points_path}')"
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_event_meta(event_id: str, data):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()

        json_data = ConvertMsg.message_to_json(data)

        sql = f"INSERT INTO event_metadata(event_id, camera_info) VALUES ('{event_id}', '{json_data}')"
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_pose_meta(data):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()

        json_data = ConvertMsg.message_to_json(data)

        sql = f"INSERT INTO pose(pose_key_points_list) VALUES ('{json_data}')"
        cur.execute(sql)
        ctx.commit()
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

    @staticmethod
    def select_autoincrement_person_id(person_id):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"SELECT id FROM people WHERE person_id = '{person_id}' AND created_at = (SELECT MAX(created_at) FROM people);"
        cur.execute(sql)
        person_id = cur.fetchone()
        ctx.close()

        return person_id[0]

    @staticmethod
    def select_autoincrement_object_id(object_id):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"SELECT id FROM object WHERE object_id = '{object_id}' AND created_at = (SELECT MAX(created_at) FROM object);"
        cur.execute(sql)
        object_id = cur.fetchone()
        ctx.close()

        return object_id[0]

    @staticmethod
    def select_autoincrement_camera_id(frame_id):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"SELECT id FROM camera WHERE name = '{frame_id}' AND created_at = (SELECT MAX(created_at) FROM camera);"
        cur.execute(sql)
        camera_id = cur.fetchone()
        ctx.close()

        return camera_id[0]
