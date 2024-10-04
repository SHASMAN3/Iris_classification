
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# Load the iris dataset
# iris = load_iris()

iris=pd.read_csv("D:\DATA SCIENCE\datascience\data\datasets\Iris_train.csv")
lb=LabelEncoder()
species=lb.fit_transform(iris["Species"])
df1=iris.drop("Species",axis=1)
iris['species']=pd.Series(species)
df=pd.concat([df1,iris['species']],axis=1)
print(lb.classes_)

# X=pd.DataFrame(iris.data,column=iris.feature_names)
# y=pd.Series(iris.target)

X=df.iloc[:,:-1]
y=df.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=0)



st.title('Iris flower classification')
from PIL import Image

image_path= r'C:\Users\Admin\Downloads\pexels-scott-platt-529822-8122418.jpg'
image =Image.open(image_path)
st.image(image,caption='Iris flower Image', use_column_width=True)
st.sidebar.header('user input parameter')
selected_model= st.sidebar.selectbox('select model',['Random Forest','SVM','KNN','Decision Tree','Logistic Regression'])



def user_input_features():
    # sepal_length=st.sidebar.slider('sepal length',float(X['sepal length (cm)'].min()),float(X['sepal length (cm)'].max()),float(X['sepal length (cm)'].mean()))
    # sepal_width=st.sidebar.slider('sepal width',float(X['sepal width (cm)'].min()),float(X['sepal width (cm)'].max()),float(X['sepal width (cm)'].mean()))
    # petal_length=st.sidebar.slider('petal length',float(X['petal length (cm)'].min()),float(X['petal length (cm)'].max()),float(X['petal length (cm)'].mean()))
    # petal_width=st.sidebar.slider('petal width',float(X['petal width (cm)'].min()),float(X['petal width (cm)'].max()),float(X['petal width (cm)'].mean()))
    sepal_length= st.number_input("sepal length", min_value=float(4.3), max_value=float(7.7), value=float(4.3), step=float(1))
    sepal_width= st.number_input("sepal width", min_value=float(2), max_value=float(4.4), value=float(2), step=float(1))
    petal_length= st.number_input("petal length", min_value=float(1), max_value=float(6.9), value=float(1), step=float(1))
    petal_width= st.number_input("petal width", min_value=float(0.10), max_value=float(2.5), value=float(0.10), step=float(1))
    
    data={'sepal length (cm)':sepal_length,
          'sepal width (cm)':sepal_width,
          'petal length (cm)':petal_length,
          'petal width (cm)':petal_width}
    
    features=pd.DataFrame(data,index=[1])
    return features


new_data= user_input_features()
model=[]
if selected_model=='Random Forest':
    model=RandomForestClassifier()
elif selected_model=='SVM':
    model=SVC()
elif selected_model=='KNN':
    model=KNeighborsClassifier()
elif selected_model=='Decision Tree':
    model=DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=2)
elif selected_model=='Logistic Regression':
    model= LogisticRegression(solver='liblinear', random_state=0)
    


    
st.write('<style>h1 {color :red;}<style >',unsafe_allow_html=True)

st.markdown('<style>div.stApp {background-color:pink;}<style >',unsafe_allow_html=True)

st.subheader('user input parameter')
st.write(new_data)

model.fit(X_train,y_train) 

prediction=model.predict(new_data)
prediction_proba=model.predict_proba(new_data)

st.subheader('class label and their corresponding index number')
# st.write(iris.target_names)
st.write(iris.Species)

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('model evaluation training')
y_pred_train=model.predict(X_train)
accuracy=accuracy_score(y_train,y_pred_train)
st.write(f'Accuracy: {accuracy:.2f}')

st.subheader('model evaluation testing')
y_pred_test=model.predict(X_test)
accuracy1=accuracy_score(y_test,y_pred_test)
st.write(f'Accuracy: {accuracy1:.2f}')



# st.subheader('Classification report training')
# st.write(classification_report(y_train,y_pred_train))

# st.subheader('Classification report testing')
# st.write(classification_report(y_train,y_pred_test))

st.write(classification_report(y_test,y_pred_test))


st.subheader('Confusion Matrix')
cm= confusion_matrix(y_test,y_pred_test)
st.write(cm)

fig, ax= plt.subplots()
scatter= ax.scatter(X_test['sepal length (cm)'],X_test['sepal width (cm)'],c=y_pred_test,cmap='viridis')
legend=ax.legend(*scatter.legend_elements(),title='Classes')
ax.add_artist(legend)
ax.set_xlabel('Sepal length (cm)')
ax.set_ylabel('Sepal width (cm)')
st.pyplot(fig)