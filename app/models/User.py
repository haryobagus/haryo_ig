from app.config import db

def get_all():
	try:
		conn = db.conn()
		with conn.cursor() as cursor:
			sql = "SELECT * FROM users ORDER BY username ASC" ## query sql
			cursor.execute(sql) ## execute query
		conn.commit()
		conn.close()
		return cursor.fetchall() ## return hasil query
	except Exception as e:
		print("Exeception occured:{}".format(e))
		return False

def get_one(id):
	try:
		conn = db.conn()
		with conn.cursor() as cursor:
			sql = "SELECT * FROM users WHERE id=%s"
			cursor.execute(sql, (id))
		conn.commit()
		conn.close()
		return cursor.fetchone()
	except Exception as e:
		print("Exeception occured:{}".format(e))
		return False

def get_by_username(username):
	try:
		conn = db.conn()
		with conn.cursor() as cursor:
			sql = "SELECT * FROM users WHERE username=%s"
			cursor.execute(sql, (username))
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
			sql = "INSERT INTO users (`username`, `password`) values (%s, %s)"
			cursor.execute(sql, (data['username'], data['password']))
		conn.commit()
		conn.close()
		return True
	except Exception as e:
		print("Exeception occured:{}".format(e))
		return False

def update(data, id):
	try:
		conn = db.conn()
		with conn.cursor() as cursor:
			old_password = data['old_password']
			if old_password != None and old_password != "":
				sql = "UPDATE users SET username=%s, password=%s WHERE id=%s"
				cursor.execute(sql, (data['username'], data['password'], id))
			else:
				sql = "UPDATE users SET username=%s WHERE id=%s"
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
			sql = "DELETE FROM users WHERE id=%s"
			cursor.execute(sql, (id))
		conn.commit()
		conn.close()
		return True
	except Exception as e:
		print("Exeception occured:{}".format(e))
		return False