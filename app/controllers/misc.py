from flask import render_template, redirect, session, url_for
from app.models import User
import bcrypt

def index():
	return render_template('pages/dashboard.html')

##AUTH
def login():
	if "user" in session:
		return redirect(url_for("index"))
	else:
		return render_template('pages/login.html')

def doLogin(data):
	user = User.get_by_username(data['username'])
	if user == None:
		return "Username tidak ditemukan"
	if user == False:
		return "Terjadi kesalahan, silahkan cek console"
	if bcrypt.checkpw(data['password'].encode('utf8'), user['password'].encode('utf8')):
		session['user'] = user
		return redirect(url_for("index"))
	else:
		return "Username atau password tidak sesuai"

def logout():
	if "user" in session:
		session.pop("user", None)

	return redirect(url_for("login"))