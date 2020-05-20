import mysql.connector
def createDatabase(cursor, database_name):
	string = "create database " + database_name
	cursor.execute(string)
def createTable(cursor, database_name):
	string = "use " + database_name
	cursor.execute(string)
	cursor.execute("create table detectedVideos (name text, path text);")

mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  passwd="123"
)

mycursor = mydb.cursor()
#createDatabase(mycursor, "detectedVideosDatabase")
#createTable(mycursor, "detectedVideosDatabase")

