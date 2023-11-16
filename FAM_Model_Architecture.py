import pickle
from tensorflow import keras
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# load data
data = pd.read_csv("HIGGS_train.csv", header=None)
X = data.drop(columns=[0])  # drop target column
y = data[0]

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# build model architecture
model = keras.Sequential([
    keras.layers.Dense(70, activation='relu', input_shape=(28,)),
    keras.layers.Dense(43, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# compile model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# define early stopping callback
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# train model
history = model.fit(X_train, y_train, epochs=50,
                    batch_size=12, validation_split=0.2)

# evaluate model on test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Accuracy:', test_acc)

# print recall, percision, f-1 score, confusion matrix
y_pred = model.predict(X_test)
y_pred = [1 if y >= 0.5 else 0 for y in y_pred]
print('Recall:', recall_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('F1 score:', f1_score(y_test, y_pred))
print('Confusion matrix:')
print(confusion_matrix(y_test, y_pred))

# plot confusion matrix
cm = confusion_matrix(y_test, y_pred)
ax = plt.subplot()
sns.heatmap(cm, annot=True, ax=ax, fmt='g')
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(['bkg', 'sig'])
ax.yaxis.set_ticklabels(['bkg', 'sig'])
plt.show()

# plot loss and accuracy
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Loss/Accuracy')
plt.legend(loc='lower right')
plt.show()

# save model
model.save("FAM_best_model.h5")

# save the model using pickle
pickle.dump(model, open("FAM_model_one.pkl", "wb"))

# load the model using pickle
FAM_model_one = pickle.load(open("FAM_model_one.pkl", "rb"))