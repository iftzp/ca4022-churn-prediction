# -*- coding: utf-8 -*-

import os
import sys
import boto3
from google.cloud import storage

# from google_drive_downloader import GoogleDriveDownloader as gdd

# sys.path.append('/usr/local/lib/python3.7/dist-packages')
# sys.path.append('/miniconda3/envs/my_env/lib/python3.6/site-packages')


os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
os.environ["SPARK_HOME"] = "spark-3.0.1-bin-hadoop2.7"   # path + xxxxxxx

# set save paths for model, extracted features df and model decision graph

file_path = "gs://ca4022-files/input/large_sparkify_event_data.json"
#file_path = "gs://ca4022-files/input/mini_sparkify_event_data.json"
model_path = "gs://ca4022-files/output/dec_tree_model"
extracted_features_path = "gs://ca4022-files/output/extracted_features_df.csv"

# import libraries
import re
import copy
import time
import random
import datetime
import numpy as np
import pandas as pd
#from matplotlib import pyplot as plt

from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml.feature import StringIndexer, VectorAssembler, OneHotEncoder
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.types import IntegerType, ArrayType, FloatType, DoubleType, Row, DateType
from pyspark.ml.linalg import DenseVector, SparseVector
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import  MulticlassClassificationEvaluator
from pyspark.ml.feature import CountVectorizer, IDF, Normalizer, PCA, RegexTokenizer, StandardScaler, StopWordsRemover, StringIndexer, VectorAssembler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', None)


from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('sparkify')\
    .config("spark.sql.broadcastTimeout", "-1").getOrCreate()



# # download large udacity dataset from AWS
# s3r = boto3.resource('s3', aws_access_key_id='AKIAIFGJ2BZJS7ELUH3Q',
#     aws_secret_access_key='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')
# buck = s3r.Bucket('udacity-dsnd')
# buck.download_file("sparkify/sparkify_event_data.json", "large_sparkify_event_data.json")



df = spark.read.json(file_path)


def cleanse_data(df):

    # cleanse stringtype fields from special characters
    for field in df.schema.fields:
        if field.dataType==StringType():
            df = df.withColumn(field.name, regexp_replace(field.name, r'[^a-zA-Z0-9\,\-]', ''))
            
    # create new ts related columns
    df = df.withColumn('interaction_time', from_unixtime(col('ts').cast(LongType())/1000).cast(TimestampType()))
    df = df.withColumn('month', month(col('interaction_time')))
    df = df.withColumn('date', from_unixtime(col('ts')/1000).cast(DateType()))

    #df = df.drop('ts')
    
    df = df.withColumn('userId', col('userId').cast(LongType()))
    # remove user id null
    df = df.filter(col('userId').isNotNull())
    # filter out log out records
    df = df = df.filter(col('auth')!='LoggedOut')
    # replace location with extracting state from it
    df = df.withColumn('location', split(col('location'),',').getItem(1))
    
    return df



def make_churn_labels(df):
   
    label_df = df.withColumn('label',
                             when((col('page').isin(['Cancellation Confirmation','Cancel'])) | (col('auth')=='Cancelled'),1 ).otherwise(0)).\
                             groupby('userId').agg(sum('label').alias('label')).withColumn('label', when(col('label')>=1 , 1).otherwise(0)
                             )

    df = df.join(label_df, on='userId')

    return df



def extract_features(df):

    # last interaction
    extracted_df =  df.groupBy('userId').agg(max('ts').alias('last_interaction'))

    extracted_df = extracted_df.join(df, on='userId').withColumn('registered_days', ((col('last_interaction')-col('registration'))/(60*1000*60*24)).cast(IntegerType()))

    # valid level
    level_df = df.orderBy('ts', ascending=False).groupBy('userId').agg(first('level').alias('valid_level'))
    
    # df = df.drop('level')
    extracted_df = extracted_df.join(level_df, on='userId')


    avg_length_df = df.groupBy('userId').avg('length').withColumnRenamed('avg(length)', 'length')
    
    # df = df.drop('length')
    extracted_df = extracted_df.join(avg_length_df, on='userId')

    daily_session_duration_df = df.groupby('userId','date','sessionId').\
    agg(max('ts').alias('session_end'), min('ts').alias('session_start')).\
    withColumn('session_duration_sec', (col('session_end')-col('session_start'))*0.001).\
    groupby('userId','date').\
    avg('session_duration_sec').\
    groupby('userId').\
    agg(mean('avg(session_duration_sec)').alias('avg_daily_session_duration')).\
    orderBy('userId', ascending=False)
    
    monthly_session_duration_df = df.groupby('userId','month','sessionId').\
    agg(max('ts').alias('session_end'), min('ts').alias('session_start')).\
    withColumn('session_duration_sec', (col('session_end')-col('session_start'))*0.001).\
    groupby('userId','month').\
    avg('session_duration_sec').\
    groupby('userId').\
    agg(mean('avg(session_duration_sec)').alias('avg_monthly_session_duration')).\
    orderBy('userId', ascending=False)
    
    duration_agg_df = daily_session_duration_df.join(monthly_session_duration_df, on='userId')
    extracted_df = extracted_df.join(duration_agg_df, on='userId')

    daily_item_df = df.groupby('userId','date').agg(max('itemInSession')).\
    groupBy('userId').avg('max(itemInSession)').\
    withColumnRenamed('avg(max(itemInSession))', 'avg_daily_items')
    
    monthly_item_df = df.groupby('userId','month').agg(max('itemInSession')).\
    groupBy('userId').avg('max(itemInSession)').\
    withColumnRenamed('avg(max(itemInSession))', 'avg_monthly_items')
    
    item_agg_df = daily_item_df.join(monthly_item_df, on='userId')
    extracted_df = extracted_df.join(item_agg_df, on='userId')
    

    listOfDistinctPages = [row.page for row in df.select('page').distinct().collect()]
    listOfDistinctPages.remove('Cancel')
    listOfDistinctPages.remove('CancellationConfirmation')
    
    daily_page_event_df = df.groupby('userId','date').pivot('page').count()
    exp_dict={}
    for page in listOfDistinctPages:
        exp_dict.update({page:'mean'})

    daily_page_event_df = daily_page_event_df.join(daily_page_event_df.groupBy('userId').agg(exp_dict).fillna(0), on='userId')

    for page in listOfDistinctPages:
        daily_page_event_df = daily_page_event_df.drop(page)  
        daily_page_event_df = daily_page_event_df.withColumnRenamed('avg({})'.format(page), 'avg_daily_{}'.format(page))

    daily_page_event_df = daily_page_event_df.drop('Cancel','CancellationConfirmation','date').drop_duplicates()
    
    
    monthly_page_event_df = df.groupby('userId','month').pivot('page').count()

    monthly_page_event_df = monthly_page_event_df.join(monthly_page_event_df.groupBy('userId').agg(exp_dict).fillna(0), on='userId')

    for page in listOfDistinctPages:
        monthly_page_event_df = monthly_page_event_df.drop(page)    
        monthly_page_event_df = monthly_page_event_df.withColumnRenamed('avg({})'.format(page), 'avg_monthly_{}'.format(page))

    monthly_page_event_df = monthly_page_event_df.drop('Cancel','CancellationConfirmation','month').drop_duplicates()

    event_agg_df = daily_page_event_df.join(monthly_page_event_df, on='userId')
    extracted_df = extracted_df.join(event_agg_df, on='userId')

    daily_session_df = df.groupby('userId','date').agg(countDistinct('sessionId')).\
    groupBy('userId').avg('count(sessionId)').\
    withColumnRenamed('avg(count(sessionId))', 'avg_daily_sessions')
    
    monthly_session_df = df.groupby('userId','month').agg(countDistinct('sessionId')).\
    groupBy('userId').avg('count(sessionId)').\
    withColumnRenamed('avg(count(sessionId))', 'avg_monthly_sessions')
    
    session_agg_df = daily_session_df.join(monthly_session_df, on='userId')
    extracted_df = extracted_df.join(session_agg_df, on='userId')

    extracted_df = extracted_df.drop('auth', 'level','length','userAgent','month','date','interaction_time','registration', 'ts','song','page','itemInSession','sessionId','artist','firstName','lastName','method','status')

    extracted_df = extracted_df.drop_duplicates()
    features = extracted_df.drop('userId')
    
    return features



def build_pipeline(num_cols):
    
    indexer_gender = StringIndexer(inputCol='gender', outputCol='gender_index', handleInvalid="skip")
    indexer_location = StringIndexer(inputCol='location', outputCol='location_index', handleInvalid="skip")
    indexer_valid_level = StringIndexer(inputCol='valid_level', outputCol='valid_level_index', handleInvalid="skip")

    assembler = VectorAssembler(inputCols=num_cols, outputCol='features', handleInvalid="skip")

    process_pipeline = Pipeline(stages=[indexer_gender, indexer_location, indexer_valid_level, assembler])

    return process_pipeline



def post_process_data(features_df):

    num_cols = []
    
    for field in features_df.schema.fields :
        if field.dataType!=StringType():
            num_cols.append(field.name)

    num_cols.remove('label')

    process_pipeline = build_pipeline(num_cols)
    model_df = process_pipeline.fit(features_df).transform(features_df)
#     return model_df.select("label", "features")
    return num_cols, model_df


from pyspark.mllib.evaluation import MulticlassMetrics

def other_eval(model, test_data):

    # Create both evaluators
    evaluatorMulti = MulticlassClassificationEvaluator(labelCol='label', predictionCol='prediction')
    evaluator = BinaryClassificationEvaluator(labelCol='label', rawPredictionCol='prediction', metricName='areaUnderROC')

    # Make predicitons
    prediction = model.transform(test_data).select('label', 'prediction')

    # Get metrics
    acc = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'accuracy'})
    f1 = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'f1'})
    weightedPrecision = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'weightedPrecision'})
    weightedRecall = evaluatorMulti.evaluate(prediction, {evaluatorMulti.metricName: 'weightedRecall'})
    auc = evaluator.evaluate(prediction)
    
    metrics = pd.DataFrame(index=['F1', 'accuracy', 'weighted precision', 'weighted recall', 'AUC'], \
                           data={'metrics value': [f1, acc, weightedPrecision, weightedRecall, auc]})
    print(metrics)
    return metrics


print("Number of rows: ")
print(df.count())


pd.DataFrame(df.take(5), columns=df.columns).head()

print("Features:\n")
df.printSchema()

print("Cleaning data...\n")
df = cleanse_data(df)

print("Creating labels from given dataFrame...\n")
df = make_churn_labels(df)

print("Extracting features...\n")
features_df = extract_features(df)


features_df.count()


pd.DataFrame(features_df.take(3), columns=features_df.columns).head()


#features_df.select([count(when(isnull(c), c)).alias(c) for c in features_df.columns]).show()

print("Extracted features:\n")
features_df.printSchema()

# save features df
print("Saved extracted features dataFrame to {}\n".format(extracted_features_path))
#features_df.write.csv(extracted_features_path)

print("dropping na values...\n")


print("Processing data for model training and testing...\n")
feats, model_data = post_process_data(features_df)


# Split the data into train, validation and test subsets
train, test = model_data.randomSplit([0.8, 0.2], seed=42)
train, validation = train.randomSplit([0.8, 0.2], seed=42)

print("Number of training examples: ")
print(train.count())


def fit_predict(train, test, model):

    mdls = {
        'logistic_regression': LogisticRegression(maxIter=10000),
        'random_forest': RandomForestClassifier(),
        'gradient_boosting': GBTClassifier(),
        'decision_tree': DecisionTreeClassifier()
    }

    if model in mdls:
        ml = mdls[model]
    else:
        return "Please choose an appropriate model"
    
    # Fit and calculate predictions
    clf = ml.fit(train)
    results = clf.transform(test)

    other_eval(clf, test)

    #evaluate_model(results)
    return clf, results


# Fit various models and visualize their accuracies WITHOUT STRATIFIED SAMPLING

# for model in ['decision_tree', 'logistic_regression', 'random_forest', 'gradient_boosting']:
for model in ['decision_tree']:
    print("\nFitting model : {}".format(model))
    clf, results = fit_predict(train, validation, model)

    print(clf)


train.groupby('label').count().toPandas()


def tune_dt(train, test, maxBins=[10, 40], maxDepth=[5, 15]):

    clf = DecisionTreeClassifier(featuresCol="features", labelCol="label")
    
    paramGrid = ParamGridBuilder() \
        .addGrid(clf.maxBins, maxBins) \
        .addGrid(clf.maxDepth, maxDepth) \
        .addGrid(clf.impurity, impurity) \
        .build()   
    
    crossval = CrossValidator(estimator = Pipeline(stages=[clf]),
                         estimatorParamMaps = paramGrid,
                         evaluator = MulticlassClassificationEvaluator(metricName='f1'),
                         numFolds = 3)

    start = time.time()

    cvModel = crossval.fit(train)
    predictions = cvModel.transform(test)
    
    #evaluate_model(predictions)

    other_eval(cvModel, test)
    
    bestPipeline = cvModel.bestModel

    # prints feature importances
    print("Most important features: (desc)\n")

    for i in range(len(bestPipeline.stages[0].featureImportances)):
        print("{} : {} \n".format(feats[i], bestPipeline.stages[0].featureImportances[i]))

    print('Best parameters : max depth:{}, num Trees:{}'.\
          format(bestPipeline.stages[0].getOrDefault('maxDepth'), bestPipeline.stages[0].getOrDefault('maxBins')))
    
    print("time elapsed: {}".format(time.time() - start))

    return bestPipeline


print("Tuning decision tree model...\n")
bestPipeline = tune_dt(train, validation)


print(bestPipeline.stages)


print(bestPipeline.stages[0].toDebugString)


print(feats)


#get the pipeline back out, as you've done earlier, this changed to [3] because of the categorical encoders
ml_pipeline = bestPipeline.stages[-1]  # was 3

#saves the model so we can get at the internals that the scala code keeps private
# model_name = "dec_mymodel_test"
ml_pipeline.write().overwrite().save(model_path)
print("Saved decision tree model to {}".format(model_path))

# read back in the model parameters
modeldf = spark.read.parquet(model_path + "/data/*")

# select only the columns that we need and collect into a list
noderows = modeldf.select("id","prediction","leftChild","rightChild","split").collect()

# create a graph for the decision tree
G = nx.Graph()

# first pass to add the nodes
for rw in noderows:

  if rw['leftChild'] < 0 and rw['rightChild'] < 0:
    if rw['prediction'] == 0:
      nc='g'
    else:
      nc='r'
    G.add_node(rw['id'], cat="Prediction", predval=rw['prediction'], disp_label=rw['prediction'], node_color=nc)
  else:
    G.add_node(rw['id'], cat="splitter", featureIndex=rw['split']['featureIndex'], thresh=rw['split']['leftCategoriesOrThreshold'], leftChild=rw['leftChild'], rightChild=rw['rightChild'], numCat=rw['split']['numCategories'], disp_label=feats[rw['split']['featureIndex']], node_color='w')

# second pass to add the relationships, now with additional information
for rw in modeldf.where("leftChild > 0 and rightChild > 0").collect():
  tempnode = G.nodes()[rw['id']]
  G.add_edge(rw['id'], rw['leftChild'], reason="{0} less than {1}".format(feats[tempnode['featureIndex']],tempnode['thresh']))
  G.add_edge(rw['id'], rw['rightChild'], reason="{0} greater than {1}".format(feats[tempnode['featureIndex']],tempnode['thresh']))

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):

    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  #allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
    
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

            
    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


import matplotlib.pyplot as plt

plt.figure(figsize=(13,13))

pos = hierarchy_pos(G, root=0, width=2)
nxlabels = nx.get_node_attributes(G, 'disp_label')
nx.draw(G, pos=pos, labels=nxlabels, with_labels=True)

gname = "gs://ca4022-files/output"
f.savefig("decision_tree_graph.png")
print("Decision Tree Graph saved in {}".format(gname))

# init GCS client and upload file
client = storage.Client()
bucket = client.get_bucket('ca4022-files')
blob = bucket.blob('output/decision_tree_graph.png')  # This defines the path where the file will be stored in the bucket
your_file_contents = blob.upload_from_filename(filename="decision_tree_graph.png")


# for x, f in enumerate(feats):
#     print(f, x)
print("Test set performance:\n")
predictions = bestPipeline.transform(test)

other_eval(bestPipeline, test)

print("\nJob Completed!")




