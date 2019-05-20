import psycopg2

x_name = ""
y_name = ""

xtrain=[]
ytrain=[]


try:
    # connect to the PostgreSQL server
    print('Connecting to the PostgreSQL database...')
    conn = psycopg2.connect(host="localhost",database="lsldb", user="lsldb_user", password="azerty")
    
    # create a cursor
    cur = conn.cursor()
        
    # execute a statement
    print('PostgreSQL database version:')
    cur.execute('SELECT * FROM ' + x_name)
    print("Retrived : ", cur.rowcount, " samples")
    row = cur.fetchone()
    while row is not None:
        print(len(row[1]))
        xtrain.append(row[1])
        row = cur.fetchone()
        
    # close the communication with the PostgreSQL
    cur.close()
except (Exception, psycopg2.DatabaseError) as error:
    print(error)
finally:
    if conn is not None:
        conn.close()
        print('Database connection closed.')

"""
x_train = np.array(x_train,float)
y_train = np.array(y_train,float)
#x_train.reshape((100,5,1))
print(x_train.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(2,2)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dense(2, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

## Recreate the exact same model, including weights and optimizer.
#model = tf.keras.models.load_model('my_model.h5')
model.compile(optimizer=tf.train.AdamOptimizer(0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
"""
