# app.py - Clasificacion de Especies de Flores Iris
# Modelos: Regresion Logistica y Red Neuronal Artificial (ANN)
# Dataset: Iris (sklearn)
# Ejecutar: python app.py

import os
import pickle
import numpy as np
# import pandas as pd (Eliminado para aligerar despliegue)
from flask import Flask, request, jsonify, render_template
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# ----- Configuracion -----
app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Nombres de las especies
SPECIES = ['Setosa', 'Versicolor', 'Virginica']
FEATURES = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']


# ----- Funciones para modelos -----

def cargar_pkl(nombre):
    """Carga un archivo .pkl si existe."""
    try:
        ruta = os.path.join(BASE_DIR, nombre)
        if os.path.exists(ruta):
            with open(ruta, 'rb') as f:
                return pickle.load(f)
    except Exception as e:
        print(f"  Error cargando {nombre}: {e}")
    return None


def entrenar_modelos():
    """Entrena ambos modelos con el dataset Iris y los guarda como .pkl."""
    print("  Entrenando modelos con dataset Iris ...")

    # Cargar datos
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Crear CSV para referencia (Deshabilitado para Vercel)
    # df = pd.DataFrame(X, columns=FEATURES)
    # df['species'] = y
    # df['species_name'] = [SPECIES[i] for i in y]
    # df.to_csv(os.path.join(BASE_DIR, 'iris.csv'), index=False)

    # Dividir en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Normalizar datos
    sc = StandardScaler()
    X_train_sc = sc.fit_transform(X_train)
    X_test_sc = sc.transform(X_test)

    # Modelo 1: Regresion Logistica
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_sc, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(X_test_sc))
    print(f"  Regresion Logistica  Accuracy: {acc_lr*100:.2f}%")

    # Modelo 2: Red Neuronal ANN
    ann = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        max_iter=500,
        random_state=42
    )
    ann.fit(X_train_sc, y_train)
    acc_ann = accuracy_score(y_test, ann.predict(X_test_sc))
    print(f"  Red Neuronal (ANN)   Accuracy: {acc_ann*100:.2f}%")

    # Guardar los 3 archivos
    # for nombre, objeto in [('logreg_model.pkl', lr), ('ann_model.pkl', ann), ('scaler.pkl', sc)]:
    #     with open(os.path.join(BASE_DIR, nombre), 'wb') as f:
    #         pickle.dump(objeto, f)

    # Guardar accuracies
    accs = {'lr': round(acc_lr * 100, 2), 'ann': round(acc_ann * 100, 2)}
    # with open(os.path.join(BASE_DIR, 'accuracies.pkl'), 'wb') as f:
    #     pickle.dump(accs, f)

    print("  Modelos guardados correctamente\n")
    return lr, ann, sc, accs


# ----- Cargar modelos al iniciar -----

# Borrar pkl viejos para reentrenar (Deshabilitado para Vercel)
# for f in ['logreg_model.pkl', 'ann_model.pkl', 'scaler.pkl', 'accuracies.pkl']:
#     ruta = os.path.join(BASE_DIR, f)
#     if os.path.exists(ruta):
#         os.remove(ruta)

logreg_model, ann_model, scaler, accuracies = entrenar_modelos()


# ----- Rutas -----

@app.route('/')
def index():
    """Pagina principal."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediccion individual de una flor."""
    data = request.get_json()
    modelo = data.get('model', 'lr')
    features = data.get('features', [])

    X = np.array(features).reshape(1, -1)
    X_sc = scaler.transform(X)

    if modelo == 'lr':
        pred = int(logreg_model.predict(X_sc)[0])
        probs = logreg_model.predict_proba(X_sc)[0].tolist()
    else:
        pred = int(ann_model.predict(X_sc)[0])
        probs = ann_model.predict_proba(X_sc)[0].tolist()

    return jsonify({
        'prediction': pred,
        'species': SPECIES[pred],
        'probabilities': {SPECIES[i]: round(p * 100, 1) for i, p in enumerate(probs)}
    })


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Prediccion por lotes."""
    data = request.get_json()
    modelo = data.get('model', 'lr')
    rows = data.get('rows', [])

    if not rows:
        return jsonify({'error': 'No se enviaron datos válidos'}), 400

    X = np.array(rows)
    if len(X.shape) > 1 and X.shape[1] > 4:
        X = X[:, :4]
    
    try:
        X_sc = scaler.transform(X)
    except Exception as e:
        return jsonify({'error': f'Error en formato de datos: {str(e)}'}), 400

    if modelo == 'lr':
        preds = logreg_model.predict(X_sc).tolist()
        probs = logreg_model.predict_proba(X_sc).tolist()
    else:
        preds = ann_model.predict(X_sc).tolist()
        probs = ann_model.predict_proba(X_sc).tolist()

    species = [SPECIES[p] for p in preds]
    prob_dicts = [{SPECIES[i]: round(pr * 100, 1) for i, pr in enumerate(row)} for row in probs]

    return jsonify({
        'predictions': preds,
        'species': species,
        'probabilities': prob_dicts
    })


@app.route('/accuracies')
def get_accuracies():
    """Devuelve accuracies de ambos modelos."""
    return jsonify(accuracies)


# ----- Iniciar servidor -----

if __name__ == '__main__':
    print("=" * 50)
    print("  IrisVision - Clasificacion de Flores Iris")
    print("=" * 50)
    print("  http://127.0.0.1:5001\n")
    app.run(debug=True, host='0.0.0.0', port=5001)
