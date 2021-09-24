from flask import render_template, redirect, url_for, jsonify
from app.models import Username

import http.client
import json

def index():
    data = Username.get_all()
    return render_template('pages/username/index.html', data=data)

def store(data):
    insert = Username.store(data)
    return redirect(url_for("username_index"))

def update(data, id):
    update = Username.update(data, id)
    return redirect(url_for("username_index"))

def delete(id):
    delete = Username.delete(id)
    return redirect(url_for("username_index"))