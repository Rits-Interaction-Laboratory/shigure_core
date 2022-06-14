import mysql.connector
import json

from shigure_core.db.config import config


class EventRepository:

    @staticmethod
    def insert_people(people_id: str, icon_path: str, icon_size: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO people(id, icon_path, icon_size) VALUES ('{people_id}', '{icon_path}', {icon_size})"
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_object(object_id: str, icon_path: str, icon_size: int):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO object(id, icon_path, icon_size) VALUES ('{object_id}', '{icon_path}', '{icon_size}')"
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
    def insert_event(event_id: str, people_id: str, object_id: str, camera_id: int, action: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO event(id, people_id, object_id, camera_id, action) VALUES ('{event_id}', '{people_id}', '{object_id}', {camera_id}, '{action}') "
        cur.execute(sql)
        ctx.commit()
        ctx.close()

    @staticmethod
    def insert_frame(event_id: str, frame_count: int, color_path: str, depth_path: str, points_path: str):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()
        sql = f"INSERT INTO frame(event_id, frame_count, color_path, depth_path, points_path) VALUES ('{event_id}', {frame_count}, '{color_path}', '{depth_path}', '{points_path}')"
        cur.execute(sql)
        ctx.close()

    @staticmethod
    def insert_event_meta(event_id: str, data):
        ctx = mysql.connector.connect(**config)
        cur = ctx.cursor()

        json_data = json.dumps(data)

        sql = f"INSERT INTO event_metadata(event_id, camera_info) VALUES ('{event_id}', '{json_data}')"
        cur.execute(sql)
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
    def show_all_tables():
        ctx = mysql.connector.connect(**config)
        cursor = ctx.cursor()

        cursor.execute("SHOW TABLES")

        for table in cursor:
            print(table[0])
        cursor.close()

    @staticmethod
    def get_all_tables():
        ctx = mysql.connector.connect(**config)
        cursor = ctx.cursor()

        cursor.execute("SHOW TABLES")

        all_tables = ()
        for table in cursor:
            all_tables = all_tables + table
        cursor.close()
        return all_tables

    @staticmethod
    def delete_all_table():
        ctx = mysql.connector.connect(**config)
        cursor = ctx.cursor()

        tables = EventRepository.get_all_tables()

        for table in tables:
            if table == "schema_migrations":
                continue
            sql = f"TRUNCATE TABLE {table};"
            cursor.execute(sql)
            ctx.commit()
        print("All data in the created table has been deleted!")
        cursor.close()



