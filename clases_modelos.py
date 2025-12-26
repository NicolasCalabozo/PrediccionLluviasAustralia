import matplotlib.pyplot as plt
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

class RedLluviaPipeline(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.001, epochs=100, batch_size=128,
                 capas_ocultas=[64, 32], dropout_rate=0.25, random_state=None):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.capas_ocultas = capas_ocultas
        self.dropout_rate = dropout_rate
        self.random_state = random_state
        
        # Estado interno
        self.best_threshold_ = 0.5
        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        # Configurar semillas para reproducibilidad
        if self.random_state is not None:
            tf.random.set_seed(self.random_state)
            np.random.seed(self.random_state)
            random.seed(self.random_state)

        input_dim = X.shape[1]
        self.classes_ = np.unique(y)

        # Split interno solo para validación (Early Stopping + Umbral)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # Pesos para desbalanceo
        pesos = compute_class_weight(class_weight='balanced', classes=self.classes_, y=y_train)
        class_weights_dict = dict(enumerate(pesos))

        # Construir y entrenar
        self._construir_modelo(input_dim)
        
        early_stopper = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6, verbose=0)

        self.model_.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            verbose=0, # Silencioso para producción
            callbacks=[early_stopper, reduce_lr],
            class_weight=class_weights_dict
        )

        # Calcular umbral óptimo
        val_probs = self.model_.predict(X_val, verbose=0)
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        # Evitar división por cero
        denominador = precision + recall
        f1_scores = np.divide(2 * precision * recall, denominador, out=np.zeros_like(precision), where=denominador!=0)
        
        mejor_idx = np.argmax(f1_scores)
        if mejor_idx < len(thresholds):
            self.best_threshold_ = thresholds[mejor_idx]
        
        return self

    def predict(self, X):
        if self.model_ is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía.")
        
        probabilidades = self.model_.predict(X, verbose=0)
        return (probabilidades > self.best_threshold_).astype(int).flatten()

    def predict_proba(self, X):
        """
        CORREGIDO: Devuelve forma (n_samples, 2) compatible con Scikit-Learn.
        Col 0: Probabilidad No Lluvia
        Col 1: Probabilidad Lluvia
        """
        if self.model_ is None:
            raise RuntimeError("El modelo no ha sido entrenado todavía.")

        # Keras da (n, 1) -> Probabilidad de Clase 1
        prob_pos = self.model_.predict(X, verbose=0).flatten()
        prob_neg = 1.0 - prob_pos
        
        # Stack para quedar (n, 2)
        return np.vstack([prob_neg, prob_pos]).T

    def _construir_modelo(self, input_dim):
        self.model_ = models.Sequential()
        for i, neuronas in enumerate(self.capas_ocultas):
            seed_capa = self.random_state + i if self.random_state else None
            if i == 0:
                self.model_.add(layers.Dense(neuronas, activation='relu', input_shape=(input_dim,)))
            else:
                self.model_.add(layers.Dense(neuronas, activation='relu'))
            self.model_.add(layers.Dropout(self.dropout_rate, seed=seed_capa))

        self.model_.add(layers.Dense(1, activation='sigmoid'))
        
        optimizador = optimizers.Adam(learning_rate=self.learning_rate)
        self.model_.compile(optimizer=optimizador, loss='binary_crossentropy', metrics=['accuracy'])

    # --- SERIALIZACIÓN (NECESARIA PARA GUARDAR EL PIPELINE ENTRENADO) ---
    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model_ is not None:
            state['model_weights_'] = self.model_.get_weights()
            state['input_dim_'] = self.model_.input_shape[1]
        # Borramos el objeto Keras (que no se puede guardar)
        if 'model_' in state: del state['model_']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_weights_' in state:
            self._construir_modelo(state['input_dim_'])
            self.model_.set_weights(state['model_weights_'])
            del self.model_weights_
            del self.input_dim_
        else:
            self.model_ = None

class HGBClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate = 0.1, max_iter = 100, max_leaf_nodes = 31, 
                 min_samples_leaf = 20, l2_regularization = 0.0, class_weight = 'balanced', 
                 early_stopping = True, validation_fraction = 0.1, scoring = 'loss',
                 n_iter_no_change = 5, random_state = 42):
        
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.l2_regularization = l2_regularization
        self.class_weight = class_weight
        self.random_state = random_state
        
        # Parámetros de control
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.scoring = scoring
        self.n_iter_no_change = n_iter_no_change
        
        self.model_ = None
        self.best_threshold_ = 0.5
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        
        # Split interno para buscar el umbral (NO para early stopping del modelo)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size = 0.2, stratify = y, random_state = self.random_state
        )
        
        self.model_ = HistGradientBoostingClassifier(
            learning_rate = self.learning_rate,
            max_iter = self.max_iter,
            max_leaf_nodes = self.max_leaf_nodes,
            min_samples_leaf = self.min_samples_leaf,
            l2_regularization = self.l2_regularization,
            class_weight = self.class_weight,
            random_state = self.random_state,
            
            # Parámetros de control
            early_stopping = self.early_stopping,
            validation_fraction = self.validation_fraction,
            scoring = self.scoring,
            n_iter_no_change = self.n_iter_no_change 
        )
        
        self.model_.fit(X_train, y_train)
        
        # Cálculo del umbral óptimo
        val_probs = self.model_.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        mejor_idx = np.argmax(f1_scores)
        
        if mejor_idx < len(thresholds):
            self.best_threshold_ = thresholds[mejor_idx]
        else:
            self.best_threshold_ = 0.5
            
        return self
    
    def predict(self, X):
        if self.model_ is None:
             raise RuntimeError("El modelo no ha sido entrenado.")
        probs = self.model_.predict_proba(X)[:, 1]
        return (probs > self.best_threshold_).astype(int)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

class RFClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', criterion = 'entropy' , bootstrap=True,
                 class_weight='balanced', n_jobs=-1, random_state=None):
        
        # Parámetros propios del Random Forest
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.criterion = criterion
        self.bootstrap = bootstrap
        self.class_weight = class_weight
        self.n_jobs = n_jobs
        self.random_state = random_state
        
        # Variables internas para el estado del modelo
        self.model_ = None
        self.best_threshold_ = 0.5
        self.classes_ = None

    def fit(self, X, y):
        # 1. Guardar clases
        self.classes_ = np.unique(y)
        
        # 2. Split interno (Solo para buscar el umbral, NO para early stopping)
        # RF no necesita early stopping, pero sí necesitamos datos 'vírgenes' para el umbral
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.random_state
        )
        
        # 3. Instanciar el modelo oficial
        self.model_ = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state
        )
        
        self.model_.fit(X_train, y_train)
        val_probs = self.model_.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-7)
        # Encontrar el índice del mejor F1
        mejor_idx = np.argmax(f1_scores)
        # Guardar el mejor umbral
        if mejor_idx < len(thresholds):
            self.best_threshold_ = thresholds[mejor_idx]
        else:
            self.best_threshold_ = 0.5

        return self

    def predict(self, X):
        """Predice usando el umbral optimizado"""
        if self.model_ is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        
        probs = self.model_.predict_proba(X)[:, 1]
        return (probs > self.best_threshold_).astype(int)

    def predict_proba(self, X):
        """Devuelve las probabilidades originales"""
        if self.model_ is None:
            raise RuntimeError("El modelo no ha sido entrenado.")
        return self.model_.predict_proba(X)