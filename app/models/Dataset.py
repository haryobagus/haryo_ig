from app.config import db

def get_all():
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = '''SELECT 
                    d.*, k.id as id_klasifikasi, k.nama as nama_klasifikasi,
                    u.id as username_id, u.username
                FROM dataset d
                INNER JOIN klasifikasi k on k.id=d.id_klasifikasi
                INNER JOIN ig_usernames u on u.id=d.id_username
                ORDER BY id DESC'''
            cursor.execute(sql)
        conn.commit()
        conn.close()
        return cursor.fetchall()
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False

def get_by_username(username):
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = '''SELECT 
                    d.*, k.id as id_klasifikasi, k.nama as nama_klasifikasi,
                    u.id as username_id, u.username
                FROM dataset d
                INNER JOIN klasifikasi k on k.id=d.id_klasifikasi
                INNER JOIN ig_usernames u on u.id=d.id_username
                WHERE u.username=%s
                ORDER BY id DESC'''
            cursor.execute(sql, (username,))
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
            sql = "SELECT * FROM dataset WHERE id=%s"
            cursor.execute(sql, (id))
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
            sql = '''INSERT INTO dataset (`id_username`, `dtime`, `tanggal_scraping`, `caption`, `casefolding`, `cleansing`, `filtering`, `stemming`, `id_klasifikasi`) 
            values (%s, %s, %s, %s, %s, %s, %s, %s, %s)'''
            cursor.execute(sql, 
                (data['id_username'], data['dtime'], data['tanggal_scraping'], data['caption'], data['casefolding'], data['cleansing'], data['filtering'], data['stemming'], data['id_klasifikasi'])
            )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False

def updateClass(id_klasifikasi, id):
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = "UPDATE dataset SET id_klasifikasi=%s WHERE id=%s"
            cursor.execute(sql, (id_klasifikasi, id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exception occured:{}".format(e))
        return False

def delete(id):
    try:
        conn = db.conn()
        with conn.cursor() as cursor:
            sql = "DELETE FROM dataset WHERE id=%s"
            cursor.execute(sql, (id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print("Exeception occured:{}".format(e))
        return False