##--Preprocessing of features for Kaggle Titanic challenge--##

# 1. Fill in missing values
# 2. Create Deck feature from Cabin feature; drop Cabin feature
# 3. Normalize numeric features
# 4. Encode categorical features

# All features provided in Kaggle data: 
# ['Age', 'Cabin', 'Embarked', 'Fare', 'Name', 'Parch', 'PassengerId',
#  'Pclass', 'Sex', 'SibSp', 'Survived', 'Ticket']

# Cabin feature is replaced with Deck feature
# Name, PassengerId, Ticket features are not used as inputs to prediction model
# Survived feature is target variable


import pandas as pd
import numpy as np                                                         
import tensorflow as tf
from tensorflow.keras.layers import Normalization, StringLookup
from tensorflow.keras.utils import to_categorical
from os import getcwd


#Get training and test set dfs
def get_dfs(train_file, test_file):
	path_train = getcwd() + '\\' + train_file
	path_test = getcwd() + '\\' + test_file

	df_train = pd.read_csv(path_train)
	df_test = pd.read_csv(path_test)

	df_train.name = 'Training Set'
	df_test.name = 'Test Set'

	return df_train, df_test

#Get count of missing values in a df
def get_missing_values(df):
	null_values = {}

	for col in df.columns.tolist():
		if df[col].isnull().sum() != 0:
			null_values[col] = df[col].isnull().sum()

	print(f'Missing values per column in {df.name} set:', null_values)

#One-hot encode categorical string features via tf.keras.layers.StringLookup
def one_hot(df):
	
	categorical_feature_names = ['Embarked', 'Sex', 'Deck']	
	vocab = {'Embarked': ['S','C','Q'], 'Sex': ['male', 'female'], 'Deck': ['M','ABC','DE','FG']}
	onehot_features = []

	#Append each one-hot encoded feature to list
	for category in categorical_feature_names:
		feature = df[category].values
		tokenizer = StringLookup(vocabulary=vocab[category], output_mode='one_hot')
		feature = tokenizer(feature)

		onehot_features.append(feature)
		
	#Delete first column of each one-hot encoded feature tensor, as it only contains zeroes 
	for i in range(len(onehot_features)):
		onehot_features[i] = np.delete(onehot_features[i], 0, axis=1)	

	return onehot_features

#Normalize numeric features
def normalize(df):
	numeric_feature_names = ['Age', 'Fare', 'Parch', 'SibSp']
	numeric_features = df[numeric_feature_names]
	numeric_features_tf = tf.convert_to_tensor(numeric_features)
	normalizer = Normalization(axis=-1)
	normalized_numeric_features = normalizer.adapt(numeric_features_tf)
	numeric_features_array = normalizer(numeric_features_tf)

	return numeric_features_array

#One-hot encode Pclass feature
def one_hot_pclass(df):
	pclass_feature = tf.convert_to_tensor(df['Pclass'])
	pclass_feature = to_categorical(pclass_feature)
	pclass_feature_final = np.delete(pclass_feature, 0, axis=1)

	return pclass_feature_final

#Engineer 'Deck' feature from 'Cabin' feature
def create_deck(df):
	#Create Deck column from first letter of Cabin column. If first letter is null, set
	#Deck = M for missing.
	df['Deck'] = df['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

	#Change deck of single passenger in T deck to A deck
	idx = df[df['Deck'] == 'T'].index
	df.loc[idx, 'Deck'] = 'A'

	#Group decks that share characteristics
	df['Deck'] = df['Deck'].replace(['A','B','C'], 'ABC')
	df['Deck'] = df['Deck'].replace(['D','E'], 'DE')
	df['Deck'] = df['Deck'].replace(['F','G'], 'FG')

	#With deck feature now replacing cabin, drop cabin feature
	df.drop(['Cabin'], inplace=True, axis=1)

	return df

#Create arrays containing processed and encoded data sets
def get_data_array(num_features, onehot_features, pclass):
	
	x = np.array(num_features)

	#onehot_features is a list; get ndarray at ith index.
	for i in range(len(onehot_features)):
		x = np.append(x, onehot_features[i], axis=1)
																			
	x = np.append(x, pclass, axis=1) 					

	return x


#Get dfs and labels
df_train, df_test = get_dfs('train.csv', 'test.csv')
y_train = df_train.pop('Survived')

#Replace 'Cabin' feature with 'Deck feature' in df_train and df_test
df_train = create_deck(df_train)
df_test = create_deck(df_test)


##--Data cleaning--##

#Replace missing age values with median according to the Sex and Pclass groups in which given 
#passenger whose age is replaced falls.
df_train_agegroup = df_train.groupby(['Sex', 'Pclass'])['Age']
df_train['Age'] = df_train_agegroup.apply(lambda x: x.fillna(x.median()))

df_test_agegroup = df_test.groupby(['Sex', 'Pclass'])['Age']
df_test['Age'] = df_test_agegroup.apply(lambda x: x.fillna(x.median()))

#Replace the two missing embarked values in training set with 'S'
df_train['Embarked'] = df_train['Embarked'].fillna('S')

#Replace the single missing fare value in test set. Get record of the passenger having missing fare value. 
#Replace according to their Pclass, Parch, and SibSp. These values for the passenger are found at [3][0][0].
missing_fare = df_test.groupby(['Pclass', 'Parch', 'SibSp']).Fare.median()[3][0][0]
df_test['Fare'] = df_test['Fare'].fillna(missing_fare)


##--Feature normalization and encoding--##

#Get normalized numeric features
numeric_features_train = normalize(df_train)
numeric_features_test = normalize(df_test)

#Get one-hot encoded features
onehot_features_train = one_hot(df_train)
onehot_features_test = one_hot(df_test)
onehot_pclass_train = one_hot_pclass(df_train)
onehot_pclass_test = one_hot_pclass(df_test)


##--Get training and test set data arrays for input to a prediction model--##
x_train = get_data_array(numeric_features_train, onehot_features_train, onehot_pclass_train)
x_test = get_data_array(numeric_features_test, onehot_features_test, onehot_pclass_test)


