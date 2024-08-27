import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from time import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
#il dataset in esame può essere caricato direttamente
# utilizzando l’utility presente in Keras
from keras.datasets import fashion_mnist

labels = ["T-shirt/top","Pantalone","Pullover","Vestito","Cappotto","Sandalo","Maglietta","Sneaker","Borsa","Stivaletto"]

# Pullover, Vestito, Cappotto, Sandalo, Maglietta, Sneaker, Borsa, Stivaletto.
# Ogni immagine è rappresentata da una matrice di 28x28 pixel,
# proviamo a visualizzare la prima osservazione del test set
# utilizzando la funzione imshow di matplotlib.

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
print("Numero totale di proprietà: "+str(X_train.shape[1]))
print("Esempi di training: "+str(X_train.shape[0]))
print("Esempi di test: "+str(X_test.shape[0]))
plt.axis("off")
plt.imshow(X_test[1], cmap="gray")
print("L'immagine figura un/o %s" % labels[y_test[1]]) #messo 1 che e'un pullover
# Ogni osservazione è una matrice, quindi dobbiamo spacchettare le righe
# all'interno di un singolo vettore utilizzando il metodo reshape.
X_train = X_train.reshape(X_train.shape[0],28*28)
X_test = X_test.reshape(X_test.shape[0],28*28)

# Ogni pixel dell'immagine ha un valore che varia
# da un valore 0 a 255
# dobbiamo dunque normalizzare tutto per ridurre
# questi valori in una scala da 0.0 a 1.0.
print("Prima della normalizzazione")
print("Valore massimo: %d" % X_train.max())
X_train = X_train/255
X_test = X_test/255

print("Dopo la normalizzazione")
print("Valore massimo: %d" % X_train.max())

#matrice di correlazione
correlation_matrix = np.corrcoef(X_train, rowvar=False)

# Plotta la heatmap della matrice di correlazione
plt.figure(figsize=(8, 6))
plt.title("Matrice di Correlazione")
plt.imshow(correlation_matrix, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Cross-Correlazione")
plt.xticks(range(len(labels)), labels, rotation=90)
plt.yticks(range(len(labels)), labels)
plt.tight_layout()
plt.show()



#Passiamo agli array con i target, ovvero quegli array
#che contengono un valore numerico da 0 a 9,
#rappresentante la categoria di appartenenza
#dell'articolo/immagine in elenco.
#Per poter eseguire una classificazione multiclasse
#dobbiamo creare 10 variabili dummy per ogni
#osservazione (una per ogni classe).
#Per procedere utilizziamo la funzione to_categorical di Keras.
from keras.utils import to_categorical
num_classes=10
y_train_dummy = to_categorical(y_train, num_classes)
y_test_dummy = to_categorical(y_test, num_classes)
# Vediamo come cambia la discesa del gradiente con la tre tipologie differenti
#Esaminiamo inizialmente la discesa del Gradiente Full batch, creando il modello
#adatto
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
# visualizziamo il numero di parametri che il Gradient Descent dovrà ottimizzare.
model.summary()

# Il risultato è oltre mezzo milione, diposto in 3 strati. Con “compile” creiamo il
#modello per l'addestramento e specifichiamo l’ottimizzatore gradient descent
model.compile(loss='categorical_crossentropy', optimizer='sgd',
metrics=['accuracy'])

# Esegui le previsioni del modello sui dati di test
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)  # Trova l'indice della classe con probabilità più alta
y_pred_pullover = y_pred_probs[:, labels.index("Pullover")]  # Prendi le probabilità per la classe "Pullover"

# Stampa le previsioni
for i in range(10):  # Stampa le previsioni per i primi 10 esempi
    print("Esempio {}: Valore Reale: {}, Valore Predetto: {}".format(i, y_test[i], y_pred[i]))

# Calcola la matrice di confusione
confusion = confusion_matrix(y_test, y_pred)


# Plotta la matrice di confusione come heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Matrice di Confusione")
plt.xlabel("Valore Predetto")
plt.ylabel("Valore Reale")
plt.show()



# Per utilizzare il full batch gradient descent,
# dobbiamo specificare all'interno del metodo fit
# in modo tale che la dimensione di un batch
# deve essere pari al numero di osservazioni
# nel set di addestramento.
# Così facendo l'algoritmo di ottimizzazione
# considererà un unico batch
# con tutte le osservazioni
# ad ogni iterazione del gradient descent.
# Per poter costruire un grafico della funzione di costo ad ogni epoca

# dobbiamo tener traccia di questi valori, per farlo dobbiamo definire un
#callback.
from keras.callbacks import History
history = History()

start_at = time()
model.fit(X_train, y_train_dummy, epochs=10, batch_size=X_train.shape[0],
callbacks=[history])
exec_time = time() - start_at
print("Tempo di addestramento: %d minuti e %d secondi" % (exec_time/60,
exec_time%60))

#I risultati del modello sono piuttosto scarsi, perché le epoche non erano
#sufficenti a portare alla convergenza, infatti con ulteriori epoche il modello
#avrebbe continuato a migliorare.
#Utilizziamo i valori della funzione di costo raccolti per visualizzare la sua
#variazione a ogni epoche su di un grafico.
plt.figure(figsize=(14,10))
plt.title("Full Batch Gradient Descent")
plt.xlabel("Epoca")
plt.ylabel("Log-Loss")
plt.plot(history.history['loss'])

# Proviamo una seconda tipologia, quella del Mini Batch Gradient Descent.
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd',
metrics=['accuracy'])

# E specifichiamo all'interno del metodo fit
# la dimensione di ogni batch, proviamo con multipli di 32 ( 32, 64, 128, 256 e
#512). Scegliamo 512.
from keras.callbacks import History
history = History()
start_at = time()

##model.fit(X_train, y_train_dummy, epochs=100, batch_size=512, callbacks=[history])
exec_time = time() - start_at
print("Tempo di addestramento: %d minuti e %d secondi" % (exec_time/60,
exec_time%60))
#come si vede dai risultati il modello e'migliorato
#Infine proviamo a regolarizzare tramite la regolarizzazione L2
print('usiamo ora la regolarizzazione L2')
from keras.regularizers import l2
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=X_train.shape[1],
kernel_regularizer=l2(0.01)))
model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer="adam", loss="categorical_crossentropy",
metrics=["accuracy"])
model.fit(X_train, y_train_dummy, epochs=100, batch_size=512)
metrics_train = model.evaluate(X_train, y_train_dummy, verbose=0)
metrics_test = model.evaluate(X_test, y_test_dummy, verbose=0)