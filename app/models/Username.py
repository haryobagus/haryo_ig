from app.config import db

def get_all():
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = '''SELECT * FROM ig_usernames ORDER BY id DESC'''
            cursor.execute(sql)
        conn.commit()
        conn.close()
        return cursor.fetchall()
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False

def get_one(id):
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = "SELECT * FROM ig_usernames WHERE id=%s"
            cursor.execute(sql, (id,))
        conn.commit()
        conn.close()
        return cursor.fetchone()
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False

def store(data):
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = '''INSERT INTO ig_usernames (`username`) values (%s)'''
            cursor.execute(sql, (data['username'],))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False

def update(data, id):
    print(data)
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = '''UPDATE ig_usernames SET `username`=%s WHERE id=%s'''
            cursor.execute(sql, (data['username'], id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False

def delete(id):
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = "DELETE FROM ig_usernames WHERE id=%s"
            cursor.execute(sql, (id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False