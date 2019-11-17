from  flask import Flask,render_template,request
import numpy
import pandas
app = Flask(__name__)
import oneoneagaindone
o = oneoneagaindone.nauman()
print("this is o",o)
'''import oneagain
p = oneagain.venkat()
print("this is o",p)
'''
#print(x,y,z)
@app.route('/') 
def home():

    #Filename
    filename1 = 'nag.xlsx'
    filename2 = 'pos.xlsx'
    filename3 = "topics.csv"

    #Read the data
    datan = pandas.read_excel(filename1, header=0)
    datap = pandas.read_excel(filename2, header=0)
    topics = pandas.read_csv(filename3, header=0)
    import xlsxwriter
    import sqlite3
    dbname = 'ava'
    conn = sqlite3.connect(dbname + '.sqlite')
    cur = conn.cursor()
    #import pandas as pd
    #if we have a csv file
    #df = pd.read_csv('avanov.csv',sep=',')
    #df.to_sql(name='avanov', con=conn)

    cur.execute('SELECT COUNT(*) FROM avanov WHERE sentiment="Strongly Negative" OR sentiment="Weekly Negative"')
    for i in cur:
        x = i
    cur.execute('SELECT COUNT(*) FROM avanov WHERE sentiment="Strongly Positive" OR sentiment="Weekly Positive" OR sentiment="Neutral"')
    for i in cur:
        y = i

    import xlsxwriter

    import sqlite3
    from xlsxwriter.workbook import Workbook
    workbook = Workbook('nag.xlsx')
    worksheet = workbook.add_worksheet()

    mysel = cur.execute('SELECT * FROM avanov WHERE sentiment="Strongly Negative" OR sentiment="Weekly Negative"')

    for i, row in enumerate(mysel):
        for j, value in enumerate(row):
            worksheet.write(i, j, value)
    workbook.close()

    from xlsxwriter.workbook import Workbook
    workboo = Workbook('pos.xlsx')
    workshee = workboo.add_worksheet()

    myself = cur.execute('SELECT * FROM avanov WHERE sentiment="Strongly Positive" OR sentiment="Weekly Positive" OR sentiment="Neutral"')

    for i, row in enumerate(myself):
        for j, value in enumerate(row):
            workshee.write(i, j, value)
    workboo.close()


    #pass into Backend
    datap = list(datap.values)
    datan = list(datan.values)
    topics = list(topics.values)


    return render_template('home.html', datap =datap,datan = datan,topics=topics,x=x[0],y=y[0],a = o[0],b=o[1],c=o[2])

app.run(debug=True)