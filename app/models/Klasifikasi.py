from app.config import db

def get_all():
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = '''SELECT * FROM klasifikasi ORDER BY id ASC'''
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
            sql = "SELECT * FROM klasifikasi WHERE id=%s"
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
            sql = '''INSERT INTO klasifikasi (`nama`) values (%s)'''
            cursor.execute(sql, (data['nama'],))
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
            sql = '''UPDATE klasifikasi SET `nama`=%s WHERE id=%s'''
            cursor.execute(sql, (data['nama'], id))
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
            sql = "DELETE FROM klasifikasi WHERE id=%s"
            cursor.execute(sql, (id,))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False