import sqlite3

conn = sqlite3.connect('testdb.sqlite')
cur = conn.cursor() # cursorインスタンスからDB操作する

# cur.execute('DROP TABLE IF EXISTS Counts') #...[4]
cur.execute('''
CREATE TABLE Vegets (veget TEXT, count INTEGER)''') #...[5]

# pieces = ["hiroki", "aaa@example.com"]
# email = pieces[1]　#...[6]
# cur.execute('SELECT count FROM Counts WHERE email = ? ', (email,)) #...[7]
# row = cur.fetchone() #...[8]
# if row is None: #...[9]
#     cur.execute('''INSERT INTO Counts (email, count)
#             VALUES (?, 1)''', (email,))
# else: #...[10]
#     cur.execute('UPDATE Counts SET count = count + 1 WHERE email = ?',
#                 (email,))
# conn.commit() #...[11]

# sqlstr = 'SELECT email, count FROM Counts ORDER BY count DESC LIMIT 10'
# for row in cur.execute(sqlstr): #...[12]
#    print(str(row[0]), row[1])
cur.close()