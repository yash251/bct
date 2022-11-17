import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv('churn_modelling.csv')

df.info()

df.head()

df.drop(columns=['RowNumber', 'CustomerId', 'Surname'], inplace=True) # cleaning

df.isna().sum()

df.describe()

# Separating the features and the labels
X=df.iloc[:, :df.shape[1]-1].values       #Independent Variables
y=df.iloc[:, -1].values                   #Dependent Variable
X.shape, y.shape

# Encoding categorical(string based) data
print(X[:8,1], '... will now become: ')
label_X_country_encoder = LabelEncoder()
X[:,1] = label_X_country_encoder.fit_transform(X[:,1])
print(X[:8,1])

print(X[:6,2], '... will now become: ')
label_X_gender_encoder = LabelEncoder()
X[:,2] = label_X_gender_encoder.fit_transform(X[:,2])
print(X[:6,2])

# Split the countries into respective dimensions. Converting the string features into their own dimensions.
transform = ColumnTransformer([("countries", OneHotEncoder(), [1])], remainder="passthrough") # 1 is the country column
X = transform.fit_transform(X)
X

# Dimensionality reduction. A 0 on two countries means that the country has to be the one variable which wasn't included
X = X[:,1:]
X.shape

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Normalize the train and test data
# ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary']
sc=StandardScaler()
X_train[:,np.array([2,4,5,6,7,10])] = sc.fit_transform(X_train[:,np.array([2,4,5,6,7,10])])
X_test[:,np.array([2,4,5,6,7,10])] = sc.transform(X_test[:,np.array([2,4,5,6,7,10])])

sc=StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
X_train

# Initialize & build the model
# INPUT = Number columns (Independet ) HIDDEN - AF HIDDEN -AF . . . N OUTPUT (1,2) -Sigmoid
from tensorflow.keras.models import Sequential
# Initializing the ANN
classifier = Sequential()

from tensorflow.keras.layers import Dense
# The amount of nodes (dimensions) in hidden layer should be the average of input and output layers, in this case 6.
# This adds the input layer (by specifying input dimension) AND the first hidden layer (units)
classifier.add(Dense(activation = 'relu', input_dim = 11, units=256, kernel_initializer='uniform'))

# Adding the hidden layer
classifier.add(Dense(activation = 'relu', units=512, kernel_initializer='uniform'))
classifier.add(Dense(activation = 'relu', units=256, kernel_initializer='uniform'))
classifier.add(Dense(activation = 'relu', units=128, kernel_initializer='uniform'))

# Adding the output layer
# Notice that we do not need to specify input dim. 
# we have an output of 1 node, which is the the desired dimensions of our output (stay with the bank or not)
# We use the sigmoid because we want probability outcomes
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform'))

# Create optimizer with default learning rate
# sgd_optimizer = tf.keras.optimizers.SGD()
# Compile the model
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

classifier.fit(
    X_train, y_train,           
    validation_data=(X_test,y_test),
    epochs=20,
    batch_size=32
)

# Predict the results using 0.5 as a threshold
y_pred = classifier.predict(X_test)
y_pred

# To use the confusion Matrix, we need to convert the probabilities that a customer will leave the bank into the form true or false. 
# So we will use the cutoff value 0.5 to indicate whether they are likely to exit or not.
y_pred = (y_pred > 0.5)
y_pred

# Print the Accuracy score and confusion matrix
from sklearn.metrics import confusion_matrix,classification_report
cm1 = confusion_matrix(y_test, y_pred)
cm1

print(classification_report(y_test, y_pred))

accuracy_model1 = ((cm1[0][0]+cm1[1][1])*100)/(cm1[0][0]+cm1[1][1]+cm1[0][1]+cm1[1][0])
print (accuracy_model1, '% of testing data was classified correctly')