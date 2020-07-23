# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 14:33:15 2020

@author: barth
"""

import pandas as pd
import sqlite3


df_init = pd.read_pickle('JREdfWithTimeInfo.pkl')

df_sql = df_init.copy()
df_sql['pod_id'] = list(df_sql.index)
col = df_sql.pop("pod_id")
df_sql.insert(0, col.name, col)


for col in ['Captions','TextSegments', 'CaptionWords', 'TextIntervalDicts']:
    df_sql[col] = df_sql[col].apply(repr)

conn = sqlite3.connect('./src/jreDB.sqlite3')
c = conn.cursor()

c.execute('CREATE TABLE JRE (id integer PRIMARY KEY NOT NULL, Title text, Description text, Views integer, \
          Rating real, Duration integer, Captions text, PodNum integer, \
          TextSegments text, CaptionWords text, Name text, TextIntervalDicts text)')
conn.commit()
    
df_sql.to_sql('JRE',conn, if_exists='replace', index=False)

# conn = sqlite3.connect('jreDB.sqlite3')
# c = conn.cursor()

# c.executescript('''
#     PRAGMA foreign_keys=off;

#     BEGIN TRANSACTION;
#     ALTER TABLE JRE RENAME TO old_table;

#     /*create a new table with the same column names and types while
#     defining a primary key for the desired column*/
#     CREATE TABLE JRE (id integer PRIMARY KEY NOT NULL, Title text, Description text, Views integer, \
#           Rating real, Duration integer, Captions text, PodNum integer, \
#           TextSegments text, CaptionWords text, Name text, TextIntervalDicts text, id integer);

#     INSERT INTO JRE SELECT * FROM old_table;

#     DROP TABLE old_table;
#     COMMIT TRANSACTION;

#     PRAGMA foreign_keys=on;''')

# #close out the connection
# c.close()
# conn.close()

# c.execute('''  
# SELECT Title FROM JRE
#           ''')
# for row in c.fetchall():
#     print (row)