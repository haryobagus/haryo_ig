import http
import json
import string
import preprocessor as p ## untuk proses cleaning

from nltk.corpus import stopwords ## library stopword
from flask import render_template, redirect, url_for, jsonify, session
from app.models import Dataset, Klasifikasi, Username
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory ## digunakan untuk proses preprocessing khususnya stemming
from datetime import datetime

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = frozenset(stopwords.words('indonesian'))

def remove_punct(text): # menghilangkan tanda baca !@#()*&^%$
    text  = "".join([char for char in text if char not in string.punctuation])
    return text

def index():
    data = Dataset.get_all()
    klasifikasi = Klasifikasi.get_all()
    usernames = Username.get_all()
    return render_template('pages/dataset/index.html', data=data, klasifikasi=klasifikasi, usernames=usernames)

def store(data):
    id_username = data["id_username"]
    count = int(data["count"])
    u = Username.get_one(int(id_username))
    username = u['username']

    conn = http.client.HTTPSConnection("instagram40.p.rapidapi.com") # menggunakan API/ proses pengambilan data
    headers = {
            'x-rapidapi-key': "7384007769msh3f9269ce3c8eae5p1b0d6fjsnf802bee62f5c",
            'x-rapidapi-host': "instagram40.p.rapidapi.com"
        }
    # kirim username ke endpoint RapidAPI, API ini akan mengambil data captions dari timeline username yang diinputkan
    conn.request("GET", "/account-feed?username={}".format(username), headers=headers) 
    res = conn.getresponse() # ambil responnnnya
    bytes_ = res.read() # baca responnya dalam bentuk bytes (0-255)
    # print(bytes_)
    today = datetime.date(datetime.now())

    captions = json.loads(bytes_.decode("utf-8")) # convert dari bytes ke json, dirubah ke json agar bisa dibaca (huruf pada umumnya)
    #print(captions,1)
    for n in range(count): # lakukan sebanyak count, misalnyna dengann 10 count, walaupun dari API dapat 100 data, proses akan berhenti setelah menndapatkan 10 caption 
        try:
            for edge in captions[n]["node"]["edge_media_to_caption"]["edges"]: # perulangan untuk setiap caption

                p.set_options(p.OPT.EMOJI,p.OPT.SMILEY) # set parameter untuk cleaning caption
                caption = p.clean(edge["node"]["text"]) # clean caption dengan menghilangkan emoji
                casefolding = caption.lower() ## lowercase
                p.set_options(p.OPT.URL,p.OPT.MENTION,p.OPT.HASHTAG,p.OPT.NUMBER,p.OPT.RESERVED) # set parameter untuk cleaning caption
                cleansing = remove_punct(p.clean(casefolding)) # menghilangkan url, mention, hashtag, angka
                
                token_caption = cleansing.split(" ") # tokenizing, mengubah string ke array
                # print(token_caption)
                # misalnya :
                # cleansing :"ini adalah sebuah caption"
                # token_caption : ["ini", "adalah", "sebuah", "caption"]
                cleaned_token = [x for x in token_caption if x not in stop_words] # menghilangkan stopwords dari array caption
                # cleaned_token : ["sebuah", "caption"]
                filtering = " ".join(cleaned_token) # untuk merubah dari array ke string lagi
                # filtering : "sebuah caption"
                stemming = stemmer.stem(filtering) # untuk menghilangkan imbuhan
                # stemming : "buah caption"

                timestamp = int(captions[n]["node"]["taken_at_timestamp"])
                dtime = datetime.fromtimestamp(timestamp)
                
                input_data = { # buat variable berisi data yang mau disimpan ke database
                    "id_username" : id_username,
                    "caption" : caption,
                    "casefolding" : casefolding,
                    "cleansing" : cleansing,
                    "filtering" : filtering,
                    "stemming" : stemming,
                    "dtime" : str(dtime),
                    "tanggal_scraping" : today,
                    "id_klasifikasi" : data["id_klasifikasi"]
                }
                Dataset.store(input_data) # memasukkan data ke database
        except IndexError: # untuk menghindari error ketika count > jumlah caption yang didapat dari API
            print("Maximum captions reached")
            break

    return redirect(url_for("dataset_index")) # redirect ke halaman dataset_index

def scrape():
    klasifikasi = Klasifikasi.get_all()
    usernames = Username.get_all()
    return render_template('pages/dataset/scraping.html', klasifikasi=klasifikasi, usernames=usernames)

def changeClass(data):
    data_id = data["data_id"]
    id_klasifikasi = data["id_klasifikasi"]
    dataset = Dataset.updateClass(id_klasifikasi, data_id)
    
    if dataset:
        response = {
            "success" : True,
            "message" : "Label berhasil diubah"
        }
    else:
        response = {
            "success" : False,
            "message" : "Gagal ketika merubah label"
        }
    return jsonify(response)

def delete(id):
    delete = Dataset.delete(id)
    return redirect(url_for("dataset_index"))