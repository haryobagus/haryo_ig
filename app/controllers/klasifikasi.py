from flask import render_template, redirect, url_for, jsonify
from app.models import Klasifikasi

import http.client
import json

def index():
    data = Klasifikasi.get_all()
    return render_template('pages/klasifikasi/index.html', data=data)

def store(data):
    insert = Klasifikasi.store(data)
    return redirect(url_for("klasifikasi_index"))

def update(data, id):
    update = Klasifikasi.update(data, id)
    return redirect(url_for("klasifikasi_index"))

def delete(id):
    delete = Klasifikasi.delete(id)
    return redirect(url_for("klasifikasi_index"))