# Databricks notebook source
import numpy as np
import pandas as pd
import tensorflow as tf 
print([tf.__version__, tf.test.is_gpu_available()])

from sklearn.model_selection import KFold
from tensorboard.plugins.hparams import api as hp

# COMMAND ----------

help(hp.RealInterval)

# COMMAND ----------

HP_NUM_UNITS = hp.HParam('num_units', hp.Discrete([128,256,512]))
HP_DROPOUT = hp.HParam('dropout', hp.Discrete([0.4]))

METRIC_ACCURACY = 'accuracy'

with tf.summary.create_file_writer('logs/hparam_tuning').as_default():
  hp.hparams_config(
    hparams=[HP_NUM_UNITS, HP_DROPOUT],
    metrics=[hp.Metric(METRIC_ACCURACY, display_name='Accuracy')],
  )

# COMMAND ----------

def train_model(hparams):
  input_shape = (28,28,1)
  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=input_shape, padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Conv2D(64, (5,5), activation='relu', padding='same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(hparams[HP_NUM_UNITS], activation='relu'),
    tf.keras.layers.Dropout(hparams[HP_DROPOUT]),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  print(model.summary())
  n_split = 4
  
  accuracy_list = []
  for train_index,test_index in KFold(n_split).split(x_data):
    x_train,x_test=x_data[train_index],x_data[test_index]
    y_train,y_test=y_data[train_index],y_data[test_index]
  
    model.fit(x_train, y_train,epochs=1, verbose=2)
  
    _, acc = model.evaluate(x_test,y_test, verbose=2)
    accuracy_list.append(acc)
    
  accuracy = np.array(accuracy_list).max() 
  return accuracy

# COMMAND ----------

def run(run_dir, hparams):
  with tf.summary.create_file_writer(run_dir).as_default():
    hp.hparams(hparams)  # record the values used in this trial
    accuracy = train_model(hparams)
    tf.summary.scalar(METRIC_ACCURACY, accuracy, step=1)

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/train.csv"
test_location = "/FileStore/tables/test.csv"
file_type = "csv"

#customSchema = StructType()

# The applied options are for CSV files. For other file types, these will be ignored.
dataDF = sqlContext.read.format(file_type).options(delimiter=',', header='true', inferschema='true').load(file_location)
testDF = sqlContext.read.format(file_type).options(delimiter=',', header='true', inferschema='true').load(test_location)

x_data = dataDF.drop('label').toPandas()
y_data = dataDF.select('label').toPandas().values

x_data = (x_data.values/255.0).reshape(-1,28,28,1)

#(validDF, trainDF) = dataDF.randomSplit([15.0, 85.0])

#x_train = trainDF.drop('label').toPandas()
#y_train = trainDF.select('label').toPandas()

#x_valid = validDF.drop('label').toPandas()
#y_valid = validDF.select('label').toPandas()

#x_train = x_train.values.reshape(-1,28,28,1)/255.0
#x_valid = x_valid.values.reshape(-1,28,28,1)/255.0

# COMMAND ----------

session_num = 18

for num_units in HP_NUM_UNITS.domain.values:
  for dropout_rate in (HP_DROPOUT.domain.values):
    hparams = {
    HP_NUM_UNITS: num_units,
    HP_DROPOUT: dropout_rate
    }
    run_name = "run-%d" % session_num
    print('--- Starting trial: %s' % run_name)
    print({h.name: hparams[h] for h in hparams})
    run('logs/hparam_tuning/' + run_name, hparams)
    session_num += 1

# COMMAND ----------

prediction = model.predict(testDF.toPandas().astype('float32').values.reshape(-1,28,28,1)/255.0)

pre_lbl = np.argmax(prediction,axis=1)
output = pd.DataFrame(pre_lbl).reset_index()
output['index'] += 1
output = output.rename({'index': 'ImageId', 0: 'Label'}, axis = 1)
outputDF = spark.createDataFrame(output)

#save prediction as csv
(outputDF
 .coalesce(1)
 .write
 .format("csv")
 .options(delimiter=',', header="true")
 .save('FileStore/submission1'))
