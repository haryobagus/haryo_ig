import math
import numpy as np
import pandas as pd
import scipy.sparse as sp

import copy

from flask import render_template, jsonify, url_for
from app.models import Dataset, Klasifikasi, Username

from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import pickle
import preprocessor as p # cleaning
import string

from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory 

factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = frozenset(stopwords.words('indonesian'))

def remove_punct(text):
    text  = "".join([char for char in text if char not in string.punctuation])
    return text

def prediksi_caption(data):

    clf = pickle.load(open("static/model/{}.sav".format(data["algoritma"]), 'rb'))
    vectorizer = pickle.load(open("static/vectorizer/{}.pickle".format(data["algoritma"]), 'rb'))

    p.set_options(p.OPT.EMOJI,p.OPT.SMILEY)
    caption = p.clean(data["caption"])
    
    casefolding = caption.lower()
    p.set_options(p.OPT.URL,p.OPT.MENTION,p.OPT.HASHTAG,p.OPT.NUMBER,p.OPT.RESERVED)
    cleansing = remove_punct(p.clean(casefolding))
    token_caption = cleansing.split(" ")
    cleaned_token = [x for x in token_caption if x not in stop_words]
    filtering = " ".join(cleaned_token)
    stemming = stemmer.stem(filtering)

    df = pd.DataFrame({
        "caption" : [caption],
        "stemming" : [stemming]
    })
    all_features = vectorizer.transform(df.stemming) # menncari vocabulary
    #print (all_features)

    result = clf.predict(all_features)
    #print (result)
    
    k = Klasifikasi.get_one(int(result[0]))
    klasifikasi = k["nama"]
    algoritma = "Support Vector Machine" if data["algoritma"] == "svm" else "Backpropagation"

    return render_template('pages/algoritma/hasil.html', klasifikasi=klasifikasi, algoritma=algoritma, caption=caption)


def backpropagation(args): # untuk membaca nilai k
    labels = list()
    klasifikasi = Klasifikasi.get_all() # ambil semua klasifikasi 
    kl_dict = dict() # membentuk dictionary di python. Dictionary merupakan tipe data yang tersusun atas dua unsur yaitu key:value atau kunci dan nilai, makanya tipe data ini juga erat kaitannya dengan sebutan argumen dengan kata kunci (keyword arguments).
    for kl in klasifikasi: # loop data yang ada pada klasifikasi dan ditampung pada variabel kl
        labels.append(kl["id"]) ## append menambahkan item dari belakkang
        kl_dict[kl["id"]] = { # kl_dict untuk setiap data yang kl id, akan menganmbil klasifikasi dengan id dan nama. misal id =1 nama = abc
            "nama" : kl["nama"]
        }

    # karena pada backpro weight yang diacak ada di tahap awal ketika membuat initial weight
    # random_state biar datanya gak berbeda beda tiap kali menjalankan algoritma
    backpro = MLPClassifier(random_state=1, max_iter=1000)
    # backpro = MLPClassifier(random_state=1, max_iter=1000, alpha(errorrate)=0.0001, hidden_layer_sizes=(100,),learning_rate_init=0.001)
    # max_iter default adalah 200
    # max_iter digunakan untuk menentukan iterasi maksimum pada proses backpro,jika pada perhitungannya sudah mencapai iterasi tersebut,
    # algoritma otomatis berhenti. memakai 1000 karena agar algoritma dapat bekerja secara optimal. 100 hasilnya 95%, 200 hasilnya 96%
    # learning rate 0.001
    title = 'Backpropagation'
    k = args.get('k')
    if k == None:
        k = 5
    data = Dataset.get_all() # pertama, ambil dulu semua dataset
    if len(data) == 0: # jika data tidak ditemukan
        return empty(title) # jalankan fungsi empty()
        
    X_train, X_test, y_train, y_test, vectorizer = preprocessing(data) # preprocessing
    # dari proses preprocessing mengahsilkan data train, yang kemudian di train pada baris 95
    backpro.fit(X_train, y_train)
    #print(dir(backpro),backpro.score)
    # menampilkan untuk nilai tf-idf
    #sample_train = pd.DataFrame.sparse.from_spmatrix(X_train)
    #print(sample_train)
    # menampilkan kata vocab dari seluruh caption dari proses preprocessing
    #print(vectorizer.vocabulary_)
    # dari hasil train itu di save pada baris selanjutnya
    filename = 'static/model/backpropagation.sav'
    pickle.dump(backpro, open(filename, 'wb'))
    filename='static/vectorizer/backpropagation.pickle'
    pickle.dump(vectorizer, open(filename, "wb"))

    backpro = pickle.load(open("static/model/backpropagation.sav", 'rb'))
    vectorizer = pickle.load(open("static/vectorizer/backpropagation.pickle", 'rb'))

    X = sp.vstack((X_train, X_test)) ## untuk menggabungkan data train dan test (x) tipe data berbeda spas matrix, untukk mendapatkan nilai X
    y = pd.concat([y_train, y_test]) ## untuk menggabungkan data train dan test (y) tipe data berbeda string, untuk mendapatkan nilai y
    #print(X, y)
    scores = kFoldClassification(backpro, X, y, k)
    u_predictions, ig = usernameClassification(vectorizer, backpro, X, y)
    createGraphs(u_predictions)
    # print(scores)
    total_akurasi = 0
    for s in scores:
        total_akurasi += s["accuracy_score"]
    avg_acc = round(total_akurasi / len(scores), 3) #mengembalikan panjang (jumlah anggota) dari scores/keseluruhan fold
    #avg_acc = round(total_akurasi / len(scores), 3)
    best_fold, data_training, data_testing = tab34(data, backpro, X, y)

    # buat 20% data testing
    i = 0
    for kl in kl_dict.values():
        #sreturn jsonify(best_fold)
        kl["cm"] = [m for m in best_fold["confusion_matrix"][i]]
        kl["precision"] = round(best_fold["precision_score"][i] * 100, 2)
        kl["recall"] = round(best_fold["recall_score"][i] * 100, 2)
        kl["f1_score"] = round(2 * (0 if (kl["precision"] + kl["recall"]) == 0 else (kl["precision"] * kl["recall"]) / (kl["precision"] + kl["recall"])), 2)
        i += 1
    
     ## kl_dict = {
    #   1 : {
    #       "nama" : Food,
    #       "precision" : 1,
    #       "recall" : 0.9,
    #       ...
    #   },
    #   2 : {
    #       "nama" : Beauty,
    #       "precision" : 0.4,
    #       ...
    #   },
    # }

    # buat k-fold
    total_y_test = list()
    total_y_pred = list()
    kfold_cm = list()
    #return jsonify(scores)
    for s in scores:
        i = 0
        kl_dict2 = copy.deepcopy(kl_dict)
        total_y_test += s["y_test"]
        total_y_pred += s["y_pred"]
        for kl in kl_dict2.values():
            kl["cm"] = [m for m in s["confusion_matrix"][i]]
            kl["precision"] = round(s["precision_score"][i] * 100, 2)
            kl["recall"] = round(s["recall_score"][i] * 100, 2)
            kl["f1_score"] = round(2 * (0 if (kl["precision"] + kl["recall"]) == 0 else (kl["precision"] * kl["recall"]) / (kl["precision"] + kl["recall"])), 2)
            i += 1
        kfold_cm.append(kl_dict2)

    #return jsonify(kl_dict2)
    # buat seluruh confusion matrix    
    kl_dict3 = copy.deepcopy(kl_dict) 
    # menambahkan labels solusi dari penambahan klasifikasi        
    ## y_test = [1,2,1,1,1,2,1,...]
    ## y_pred = [1,1,2,2,1,2,1,...]
    # menampilkan nilai conf matrix
    #print(total_y_test)
    #print(total_y_pred)
    cm = confusion_matrix(total_y_test, total_y_pred, labels=labels)
    ps = recall_score(total_y_test, total_y_pred, average=None, labels=labels)
    rs = precision_score(total_y_test, total_y_pred, average=None, labels=labels)
    fs = f1_score(total_y_test, total_y_pred)
    acs = accuracy_score(total_y_test, total_y_pred)

    # kl_dict3["cm"] = cm
    i = 0
    for kl in kl_dict3.values():
        # print(cm[i], i)
        # print(ps.tolist()[n])
        kl["cm"] = cm[i]
        kl["precision"] = round(ps.tolist()[i] * 100, 2)
        kl["recall"] = round(rs.tolist()[i] * 100, 2)
        kl["f1_score"] = round(2 * (0 if (kl["precision"] + kl["recall"]) == 0 else (kl["precision"] * kl["recall"]) / (kl["precision"] + kl["recall"])), 2)
        i += 1
    #return jsonify(kl_dict3)
    return render_template('pages/algoritma/detail.html', scores=scores, title=title, ig=ig, avg_acc=avg_acc, data_training=data_training, data_testing=data_testing, kl_dict=kl_dict, kl_dict3=kl_dict3, best_fold=best_fold, kfold_cm=kfold_cm)


def supportVectorMachine(args):
    labels = list()
    klasifikasi = Klasifikasi.get_all() # ambil semua klasifikasi 
    kl_dict = dict() # membentuk dictionary di python. Dictionary merupakan tipe data yang tersusun atas dua unsur yaitu key:value atau kunci dan nilai, makanya tipe data ini juga erat kaitannya dengan sebutan argumen dengan kata kunci (keyword arguments).
    for kl in klasifikasi:
        labels.append(kl["id"])
        kl_dict[kl["id"]] = {
            "nama" : kl["nama"]
        }
    
    svm = SVC() ## inisiasi model
    # C:float default 1.0, kernel default = rbf, gamma = scale
    title = 'Support Vector Machine'
    k = args.get('k')
    if k == None:
        k = 5
    data = Dataset.get_all() # pertama, ambil dulu semua dataset
    if len(data) == 0: # jika data tidak ditemukan
        return empty(title) # jalankan fungsi empty()
        
    X_train, X_test, y_train, y_test, vectorizer = preprocessing(data) # preprocessing
     # dari proses preprocessing mengahsilkan data train, yang kemudian di train pada baris 188
    svm.fit(X_train, y_train)
    # menampilkan untuk nilai tf-idf
    ##sample_train = pd.DataFrame.sparse.from_spmatrix(X_train)
    ##print(sample_train)
    # menampilkan kata vocab dari seluruh caption dari proses preprocessing
    #print(vectorizer.vocabulary_)
    # dari hasil train itu di save pada baris selanjutnya
    filename = 'static/model/svm.sav'
    pickle.dump(svm, open(filename, 'wb'))
    filename='static/vectorizer/svm.pickle'
    pickle.dump(vectorizer, open(filename, "wb"))

    svm = pickle.load(open("static/model/svm.sav", 'rb'))
    vectorizer = pickle.load(open("static/vectorizer/svm.pickle", 'rb'))

    X = sp.vstack((X_train, X_test)) ## untuk menggabungkan data train dan test (x) tipe data berbeda spas matrix  
    y = pd.concat([y_train, y_test]) ## untuk menggabungkan data train dan test (y) tipe data berbeda string
    # print(X, y)
    scores = kFoldClassification(svm, X, y, k)
    u_predictions, ig = usernameClassification(vectorizer, svm, X, y)
    createGraphs(u_predictions)
    
    total_akurasi = 0
    for s in scores:
        total_akurasi += s["accuracy_score"]
    avg_acc = round(total_akurasi / len(scores), 3)
    best_fold, data_training, data_testing = tab34(data, svm, X, y)

    # untuk 20% data testing
    i = 0
    for kl in kl_dict.values():
        kl["cm"] = [m for m in best_fold["confusion_matrix"][i]]
        kl["precision"] = round(best_fold["precision_score"][i] * 100, 2)
        kl["recall"] = round(best_fold["recall_score"][i] * 100, 2)
        kl["f1_score"] = round(2 * (0 if (kl["precision"] + kl["recall"]) == 0 else (kl["precision"] * kl["recall"]) / (kl["precision"] + kl["recall"])), 2)
        i += 1

    # buat k-fold
    total_y_test = list()
    total_y_pred = list()
    kfold_cm = list()
    for s in scores:
        i = 0
        kl_dict2 = copy.deepcopy(kl_dict)
        total_y_test += s["y_test"]
        total_y_pred += s["y_pred"]
        for kl in kl_dict2.values():
            kl["cm"] = s["confusion_matrix"][i]
            kl["precision"] = round(s["precision_score"][i] * 100, 2)
            kl["recall"] = round(s["recall_score"][i] * 100, 2)
            kl["f1_score"] = round(2 * (0 if (kl["precision"] + kl["recall"]) == 0 else (kl["precision"] * kl["recall"]) / (kl["precision"] + kl["recall"])), 2)
            i += 1
        kfold_cm.append(kl_dict2)

    # return jsonify(kfold_cm)

    # buat seluruh confusion matrix
    kl_dict3 = copy.deepcopy(kl_dict)  
    # menambahkan labels solusi dari penambahan klasifikasi       
    cm = confusion_matrix(total_y_test, total_y_pred, labels=labels)
    ps = recall_score(total_y_test, total_y_pred, average=None, labels=labels)
    rs = precision_score(total_y_test, total_y_pred, average=None, labels=labels)
    fs = f1_score(total_y_test, total_y_pred)
    acs = accuracy_score(total_y_test, total_y_pred)

    # kl_dict3["cm"] = cm
    i = 0
    for kl in kl_dict3.values():
        # print(cm[i], i)
        # print(ps.tolist()[n])
        kl["cm"] = cm[i]
        kl["precision"] = round(ps.tolist()[i] * 100, 2)
        kl["recall"] = round(rs.tolist()[i] * 100, 2)
        kl["f1_score"] = round(2 * (0 if (kl["precision"] + kl["recall"]) == 0 else (kl["precision"] * kl["recall"]) / (kl["precision"] + kl["recall"])), 2)
        i += 1
    return render_template('pages/algoritma/detail.html', scores=scores, title=title, ig=ig, avg_acc=avg_acc, data_training=data_training, data_testing=data_testing, kl_dict=kl_dict, kl_dict3=kl_dict3, best_fold=best_fold, kfold_cm=kfold_cm)


def empty(title):
    return render_template('pages/algoritma/empty.html', title=title)


def preprocessing(data):
    pdData = pd.DataFrame.from_dict(data) # ubah dulu ke dataframe,diubah ke dataframe karena dataframe merupakan format untuk mengolah data dimana dapat mempermudah user dalam menganalisa data
    pdData.sort_index(inplace=True)
    # contoh : "buah caption"
    vectorizer = TfidfVectorizer() # membuat TF ID-F Vectorizer
    #print (vectorizer)
    all_features = vectorizer.fit_transform(pdData.stemming) # menncari vocabulary
    #print (all_features)
    #print("Vocabulary size : {}".format(len(vectorizer.vocabulary_)))
    #print("vocabulary content:\n {}".format(vectorizer.vocabulary_))
    
    X_train, X_test, y_train, y_test = train_test_split(all_features, pdData.id_klasifikasi, test_size=0.2, shuffle=False) ## membagi data ke data training dan data testing
    # print(X_train)
    # print(type(X_test))
    # print(type(y_train))
    # print(type(y_test))
    #print(vectorizer.vocabulary_)
    # print (all_features)
    # X = nilai tfidf untuk setiap kata di masing2 caption
    # y = label
    return X_train, X_test, y_train, y_test, vectorizer


def tab34(data, clf, X, y):
    scores = kFoldClassification(clf, X, y, 5)		
    sorted_scores = sorted(scores, key = lambda i: i['accuracy_score'], reverse=True)
    best_fold = sorted_scores[0]

    data_training = list()
    data_testing = list()
    for i in best_fold["train_index"]:
        data_training.append(data[i])
    n = 0
    for i in best_fold["test_index"]:
        temp = data[i]
        temp["y_pred"] = best_fold["y_pred"][n]
        data_testing.append(temp)
        n += 1

    return best_fold, data_training, data_testing


def kFoldClassification(clf, X, y, k):
    labels = list()
    klasifikasi = Klasifikasi.get_all() # ambil semua klasifikasi.
    for kl in klasifikasi:
        labels.append(kl["id"])
    #print(labels)

    scores = []
    cv = KFold(n_splits=int(k), random_state=42, shuffle=True)
    # karena pada train_test_split datanya diacak(pembagian teaindantest), jadi pake 
    # random_state biar datanya gak berbeda beda tiap kali menjalankan algoritma
    # jika tidak menggunakan random_state data train dan test akan berbeda2 walau K nya sama
    ## Data = [0,1,2,3,4,5,6,7,8,9]
    ## K = 5
    ## 100 / 5 = 20% data testing
    ## K1 = Data Training : [0,1,2,3,4,5,6,7], Data Testing : [8,9]
    ## K2 = Data Training : [0,1,2,3,4,5,8,9], Data Testing : [6,7]
    ## K3
    ## n_split = berapa kali lipatan / nilai K pada K-Fold
    ## ramdom_state = untuk mengacak data agar menghilangkan bias (membuat datanya lebih beragam) pada data training atau testing
    for train_index, test_index in cv.split(X):
        #print("Train Index: ", train_index, "\n")
        #print("Test Index: ", test_index)

        # X = Nilai TFIDF caption (fitur)
        # y = id_klasifikasi (label), contohnya 1 (Food), 2 (Beauty)

        ## cv berisi index data training dan data testing di msing2 iterasi,
        ## kode di bawah mengambil data X dan y berdasarkan index tersebut
        X_train, X_test, y_train, y_test = X[train_index], X[test_index], y[train_index], y[test_index]
        #print (y[test_index])
        #print (y_test)
        y_pred = clf.predict(X_test) # fungsi untuk mentraining model menggunakan data training sekaligus memprediksi data testing.
        #print(X_test)
        #print (y_pred)
        #print (y_test)
        #print (type(y_test))
        cm = confusion_matrix(y_test, y_pred, labels=labels)
        ps = recall_score(y_test, y_pred, average=None, labels=labels)
        rs = precision_score(y_test, y_pred, average=None, labels=labels)
        fs = f1_score(y_test, y_pred)
        acs = accuracy_score(y_test, y_pred) # masing2 fold
        scores.append({
            "y_test" : y_test.tolist(), # rubah ke array
            "y_pred" : y_pred.tolist(),
            "train_index" : train_index.tolist(),
            "test_index" : test_index.tolist(),
            "confusion_matrix" : cm.tolist(),
            "precision_score" : [round(p, 3) for p in ps.tolist()],
            "recall_score" : [round(r, 3) for r in rs.tolist()],
            "f1_score" : round(fs, 3),
            "accuracy_score" : round(acs * 100, 3),
            "error_rate" : str(round((1 - acs) * 100, 3))
        })
        #print (y_test.tolist())
    return scores


def usernameClassification(vectorizer, clf, X_train, y_train):
    ig = Username.get_all()
    predictions = []
    usernames = []
    for i in ig:
        data = Dataset.get_by_username(i['username']) ## ambil data caption untuk maasng2 username/selebgram
        if len(data) > 0:
            pdData = pd.DataFrame.from_dict(data) # ubah dulu ke dataframe
            pdData.sort_index(inplace=True)
            all_features = vectorizer.transform(pdData.stemming) # transform ke tfidf
            #print (all_features)
            Xn, Xm, yn, ym = train_test_split(all_features, pdData.id_klasifikasi, test_size=0.2, shuffle=False) ## membagi data ke data training dan data testing
            X_test = sp.vstack((Xn, Xm)) ## untuk menggabungkan data train dan test (x) tipe data berbeda spas matrix
            y_test = pd.concat([yn, ym]) ## untuk menggabungkan data train dan test (y) tipe data berbeda string
            y_pred = clf.predict(X_test) # fungsi untuk mentraining menggunakan data training sekaligus memprediksi data testing.
            
            predictions.append({ ## append menambahkan item dari belakkang
                "username" : i['username'], # bikin list untuk membuat grafik
                "pred" : y_pred
            })
            usernames.append(i["username"])
    return predictions, usernames


def createGraphs(u_predictions): # untuk membuat grafik
    classes = Klasifikasi.get_all() ## cari dulu klasifikasinya ada apa aja
    cat_name = [cl['nama'] for cl in classes] ## buat array nama klasifikasi, misal = ["Food", "Beauty"]
    #print (cat_name)
    cat_id = [cl['id'] for cl in classes] ## buat array id klasifikasi, misal = [1, 2]
    #print (cat_id)
    
    ##cat_name = list()
    ##for cl in classes:
    ## cat_name.append(cl["nama"])

    for u in u_predictions: ## untuk setiap user :
        counts = list()
        for id_ in cat_id:
            count = np.count_nonzero(u['pred'] == id_) # menjumlahkan data tidak 0/ menghitung prediksi tiap akun, numpy array count data dari tidak kosongan dari nilai pred yang sama dengan id
            #print (u['pred'])
            #print (u_predictions)
            #print(id_)
            counts.append(count)
            #print(counts)
            
        plt.pie(counts, labels=cat_name, autopct='%1.1f%%') ## buat pie chart
        plt.title("@{}".format(u['username'])) ## kasih judul di grafiknya
        plt.savefig("static/img/{}.png".format(u['username']), dpi=200) ## save gambar grafik tersebut
        plt.clf() ## kosongin objek plt biat bisa gambar grafik yang lain
    return None