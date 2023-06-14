import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simular datos de características de proteínas (X) y etiquetas de interacciones (y)
n_instances = 1000  # Número total de instancias
n_features = 20  # Número de características por instancia

# Generar características aleatorias para las proteínas
X = np.random.rand(n_instances, n_features)

# Generar etiquetas de interacciones (1 para interacción, 0 para no interacción)
y = np.random.randint(2, size=n_instances)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de clasificación (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
