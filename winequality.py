import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

#load dataset
df=pd.read_csv(r"C:\Users\DELL\Downloads\wine+quality\winequality-red.csv",sep=";") #path of winequality dataset file
#print(df.head())

#checking duplicates in dataset
df.duplicated().sum() 

#biary classification
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 6 else 0) #Binary classification  1 if quality >= 6 (Good), else 0 (Bad)

#dropping quality because we are using binary 
df.drop("quality", axis = 1, inplace=True) #drop quality column

#define x feature and y target
features = ['alcohol', 'sulphates', 'volatile acidity', 'citric acid', 'pH', 'density', 'residual sugar','chlorides', 'free sulfur dioxide','total sulfur dioxide']
x = df[features]#important features for prediction
y = df['quality_label']#target

#feature scaling normalization
scaler = MinMaxScaler()
xscaled = scaler.fit_transform(x)

#model training x y train test split
x_train, x_test, y_train, y_test = train_test_split(xscaled, y, test_size=0.2, random_state=42)

 #hyperparameter grid for tuning
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

#using grid search 
grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=5) #grid search
grid.fit(x_train, y_train)
model = grid.best_estimator_  #getting best model for tuning
print("Best Parameters:", grid.best_params_)

# Evaluation
y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

 #accuracy of training and testing 
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

#accuracy of training and testing dsiplay in app
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

#classification report disply in app
print("\n Classification Report:\n")
print(classification_report(y_test, y_test_pred, target_names=['Bad', 'Good']))

#confusion matrix or checking bad and good quality  #optional
confusionm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=confusionm, display_labels=['Bad', 'Good'])
disp.plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.grid(False)
plt.tight_layout()
plt.show()

#plot graph of feature importance
importances = model.feature_importances_
sns.barplot(x=importances, y=features)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.grid(True)
plt.tight_layout()
plt.show()

# input from  user or cli
print("\n Enter wine properties: ")
userinput=[]
for feature in features:
    val=float(input(f"Enter the {feature} value: "))
    userinput.append(val)

## Preprocess input and predict data
userscaled = scaler.transform([userinput])
prediction = model.predict(userscaled)[0]

##prediction on the basis of input data
quality = "Good" if prediction == 1 else "Bad"
print(f"\n Based on the input values, the predicted wine quality is: {quality}")
