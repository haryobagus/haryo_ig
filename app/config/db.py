import pymysql.cursors

# Connect to the database

# host='localhost' 
# user='haryoig'
# password='haryoig'
# database='haryoig'

host='localhost'
user='root'
password=''
database='aplikasiskripsi_haryo_ig'


def conn():
    conn = pymysql.connect(host=host, user=user, password=password, database=database, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    return conn