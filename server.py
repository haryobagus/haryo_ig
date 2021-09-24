from flask_cors import CORS
from flask import Flask, render_template, request
from app.config.middleware import checkLogin
from app.controllers import misc, user, dataset, klasifikasi, algoritma, username
import os

## di atas adalah modul2 yang digunakan

app = Flask(__name__)
CORS(app)

##Support Vector Machine
@app.route("/algoritma/svm")
@checkLogin  ## fungsinya untuk ngecek apakah pengguna sudah login atau belum
def svm_index():
    return algoritma.supportVectorMachine(request.args)

##Backpropagation
@app.route("/algoritma/backpro")
@checkLogin
def backpro_index():
    return algoritma.backpropagation(request.args)

##crud
##Pengguna
@app.route("/users")
@checkLogin
def user_index():
    return user.index() ## ini berarti menjalankan fungsi index yang ada di modul user. Modulnya bisa dicari di atas (import)

##methods : 
# GET = ngambil data
# POST = nyimpen data
# PUT = update data
# DELETE = hapus data

@app.route("/user/store", methods=['POST'])
@checkLogin
def user_store():
    return user.store(request.form)

@app.route("/user/<int:id>/update", methods=['POST'])
@checkLogin
def user_update(id):
    return user.update(request.form, id)

@app.route("/user/<int:id>/delete", methods=['POST'])
@checkLogin
def user_delete(id):
    return user.delete(id)

##klasifikasi
@app.route("/klasifikasi")
@checkLogin
def klasifikasi_index():
    return klasifikasi.index()

@app.route("/klasifikasi/store", methods=['POST'])
@checkLogin
def klasifikasi_store():
    return klasifikasi.store(request.form)

@app.route("/klasifikasi/<int:id>/update", methods=['POST', 'PUT'])
@checkLogin
def klasifikasi_update(id):
    return klasifikasi.update(request.form, id)

@app.route("/klasifikasi/<int:id>/delete", methods=['POST', 'DELETE'])
@checkLogin
def klasifikasi_delete(id):
    return klasifikasi.delete(id)

##username
@app.route("/username")
@checkLogin
def username_index():
    return username.index()

@app.route("/username/store", methods=['POST'])
@checkLogin
def username_store():
    return username.store(request.form)

@app.route("/username/<int:id>/update", methods=['POST', 'PUT'])
@checkLogin
def username_update(id):
    return username.update(request.form, id)

@app.route("/username/<int:id>/delete", methods=['POST', 'DELETE'])
@checkLogin
def username_delete(id):
    return username.delete(id)


##Dataset
@app.route("/dataset")
@checkLogin
def dataset_index():
    return dataset.index()

@app.route("/dataset/tanggal-scraping", methods=['GET'])
@checkLogin
def dataset_tanggal():
    return dataset.updateTanggal()

@app.route("/dataset/scrape", methods=['GET'])
@checkLogin
def dataset_scrape():
    return dataset.scrape()

@app.route("/dataset/store", methods=['POST'])
@checkLogin
def dataset_store():
    return dataset.store(request.form)

@app.route("/dataset/change-class", methods=["GET"])
@checkLogin
def dataset_change_class():
    return dataset.changeClass(request.args)

@app.route("/dataset/<int:id>/delete", methods=['POST', 'DELETE'])
@checkLogin
def dataset_delete(id):
    return dataset.delete(id)

##MISC
@app.route("/")
def index():
	return misc.index()

##MISC
@app.route("/input-caption", methods=["POST"])
def input_caption():
	return algoritma.prediksi_caption(request.form)

@app.route("/login")
def login():
	return misc.login()

@app.route("/doLogin", methods=['POST'])
def doLogin():
	return misc.doLogin(request.form)

@app.route("/logout")
def logout():
	return misc.logout()

app.secret_key = '3RDLwwtFttGSxkaDHyFTmvGytBJ2MxWT8ynWm2y79G8jm9ugYxFFDPdHcBBnHp6E'
app.config['SESSION_TYPE'] = 'filesystem'

@app.context_processor
def inject_stage_and_region():
	return dict(APP_NAME=os.environ.get("APP_NAME"),
		APP_AUTHOR=os.environ.get("APP_AUTHOR"),
		APP_TITLE=os.environ.get("APP_TITLE"),
		APP_LOGO=os.environ.get("APP_LOGO"))

if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5299)