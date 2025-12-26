import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import wkt
from sklearn.preprocessing import FunctionTransformer
from sklearn.base import BaseEstimator, TransformerMixin

class AgregadorEspacial(BaseEstimator, TransformerMixin):
    def __init__(self, shapefile_path):
        self.shapefile_path = shapefile_path
        self.gdf_regiones = None
        self.stations_points = {
            "Albury": "POINT(146.91583 -36.08056)",
            "BadgerysCreek": "POINT(150.75222 -33.87972)",
            "Cobar": "POINT(145.83194 -31.49972)",
            "CoffsHarbour": "POINT(153.11889 -30.30222)",
            "Moree": "POINT(149.83389 -29.46583)",
            "Newcastle": "POINT(151.75000 -32.91700)",
            "NorahHead": "POINT(151.57417 -33.28250)",
            "NorfolkIsland": "POINT(167.951564 -29.033794 )",
            "Penrith": "POINT(150.69450 -33.75150)",
            "Richmond": "POINT(150.78400 -33.58600)",
            "Sydney": "POINT(151.21000 -33.86778)",
            "SydneyAirport": "POINT(151.17722 -33.94611)",
            "WaggaWagga": "POINT(147.35900 -35.10900)",
            "Williamtown": "POINT(151.83444 -32.79500)",
            "Wollongong": "POINT(150.88300 -34.41700)",
            "Canberra": "POINT(149.12694 -35.29306)",
            "Tuggeranong": "POINT(149.08600 -35.40900)",
            "MountGinini": "POINT(148.95000 -35.47000)",
            "Ballarat": "POINT(143.85000 -37.55000)",
            "Bendigo": "POINT(144.28278 -36.75917)",
            "Sale": "POINT(147.05400 -38.10340)",
            "MelbourneAirport": "POINT(144.84479 -37.66371)",
            "Melbourne": "POINT(144.96306 -37.81361)",
            "Mildura": "POINT(142.15833 -34.18889)",
            "Nhil": "POINT(141.64722 -36.31083)",
            "Portland": "POINT(141.47111 -38.31806)",
            "Watsonia": "POINT(145.08300 -37.70800)",
            "Dartmoor": "POINT(141.28333 -37.93333)",
            "Brisbane": "POINT(153.02806 -27.46778)",
            "Cairns": "POINT(145.77330 -16.92330)",
            "GoldCoast": "POINT(153.40000 -28.01667)",
            "Townsville": "POINT(146.81580 -19.26220)",
            "Adelaide": "POINT(138.60072 -34.92866)",
            "MountGambier": "POINT(140.63444 -37.81028)",
            "Nuriootpa": "POINT(138.86500 -34.50100)",
            "Woomera": "POINT(136.81694 -31.14417)",
            "Albany": "POINT(117.88139 -35.02278)",
            "Witchcliffe": "POINT(115.09900 -34.02400)",
            "PearceRAAF": "POINT(116.01500 -31.66700)",
            "PerthAirport": "POINT(115.96700 -31.94030)",
            "Perth": "POINT(115.86000 -31.95000)",
            "SalmonGums": "POINT(121.64500 -32.98000)",
            "Walpole": "POINT(116.72300 -34.93100)",
            "Hobart": "POINT(147.32500 -42.88055)",
            "Launceston": "POINT(147.30440 -41.36050)",
            "AliceSprings": "POINT(133.87000 -23.70000)",
            "Darwin": "POINT(130.84111 -12.43806)",
            "Katherine": "POINT(132.26667 -14.46667)",
            "Uluru": "POINT(131.03611 -25.34500)"
        }
        
    def fit(self, X, y=None):
        self.gdf_regiones = gpd.read_file(self.shapefile_path)

        if self.gdf_regiones.crs is None:
             self.gdf_regiones.set_crs(epsg = 4326, inplace = True)
             
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        X_copy['geometry'] = X_copy['Location'].map(self.stations_points).apply(wkt.loads)
        clima_gdf = gpd.GeoDataFrame(X_copy, geometry = 'geometry', crs = "EPSG:4326")

        if self.gdf_regiones.crs != clima_gdf.crs:
            self.gdf_regiones = self.gdf_regiones.to_crs(clima_gdf.crs)

        estaciones_regiones = gpd.sjoin(clima_gdf, self.gdf_regiones, how='left', predicate='within')
        columnas_a_borrar = ['index_right', 'OBJECTID', 'Shape_Leng', 'Shape_Area', 'code', 'geometry']
        cols_existentes = [c for c in columnas_a_borrar if c in estaciones_regiones.columns]
        estaciones_regiones = estaciones_regiones.drop(columns=cols_existentes)
        estaciones_regiones.loc[estaciones_regiones['Location'] == "NorfolkIsland", 'label'] = 'Offshore'

        return pd.DataFrame(estaciones_regiones)
    
        

class ImputadorNumerico(BaseEstimator, TransformerMixin):
    def __init__(self, variables_imputar, variables_contexto):
        self.variables_imputar = variables_imputar
        self.variables_contexto = variables_contexto

    def fit(self, X, y=None):
        self.mediana_global_ = X[self.variables_imputar].median()
        self.medianas_region_ = X.groupby(self.variables_contexto)[self.variables_imputar].median()
        return self

    def transform(self, X):
        X_copia = X.copy()
        X_con_medianas = X_copia.join(
            self.medianas_region_, 
            on=self.variables_contexto, 
            rsuffix='_med'
        )
        for col in self.variables_imputar:
            col_mediana = f"{col}_med"
            X_copia[col] = X_copia[col].fillna(X_con_medianas[col_mediana])
            X_copia[col] = X_copia[col].fillna(self.mediana_global_[col])
            
        return X_copia[self.variables_imputar]
    
    def get_feature_names_out(self, input_features=None):
        # Devolvemos solo las columnas que realmente salen del transformador
        return self.variables_imputar

class ImputadorCategorico(BaseEstimator, TransformerMixin):
    def __init__(self, seed=13):
        self.seed = seed

    def fit(self, X, y=None):
        # Guardamos los nombres de las columnas que vimos al entrenar
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
            
        self.mapa_frecuencias_ = {}
        for columna in X.columns:
            self.mapa_frecuencias_[columna] = X[columna].value_counts(normalize=True)
        return self

    def transform(self, X):
        X_copia = X.copy()
        rng = np.random.default_rng(self.seed)

        for columna in X.columns:
            if columna in self.mapa_frecuencias_:
                frecuencias = self.mapa_frecuencias_[columna]
                mask_nulos = X_copia[columna].isnull()
                num_nulos = mask_nulos.sum()
                if num_nulos > 0:
                    valores_imputados = rng.choice(
                        a=frecuencias.index,
                        size=num_nulos,
                        p=frecuencias.values
                    )
                    X_copia.loc[mask_nulos, columna] = valores_imputados
        return X_copia

    def get_feature_names_out(self, input_features=None):
        # Si tenemos los nombres guardados, los usamos. Si no, usamos input_features.
        if hasattr(self, "feature_names_in_"):
            return self.feature_names_in_
        if input_features is not None:
            return input_features
        return None

class CodificadorCiclico(BaseEstimator, TransformerMixin):
    def __init__(self, categorias_ordenadas, drop_original=True):
        self.categorias_ordenadas = categorias_ordenadas
        self.drop_original = drop_original

    def fit(self, X, y=None):
        # Guardamos qué columnas entraron para saber a cuáles ponerles _sin/_cos
        if hasattr(X, "columns"):
            self.feature_names_in_ = list(X.columns)
        return self

    def transform(self, X):
        X_copia = X.copy()
        mapeo = {cat: i for i, cat in enumerate(self.categorias_ordenadas)}
        max_val = len(self.categorias_ordenadas)

        for columna in X.columns:
            serie_mapeada = X_copia[columna].map(mapeo)
            X_copia[f"{columna}_sin"] = np.sin(2 * np.pi * serie_mapeada / max_val)
            X_copia[f"{columna}_cos"] = np.cos(2 * np.pi * serie_mapeada / max_val)
            
            if self.drop_original:
                X_copia.drop(columns=[columna], inplace=True)

        return X_copia
    
    # --- CORRECCIÓN AQUÍ ---
    def get_feature_names_out(self, input_features=None):
        # Obtenemos los nombres de entrada
        if input_features is not None:
            names = input_features
        elif hasattr(self, "feature_names_in_"):
            names = self.feature_names_in_
        else:
            return None # O lanzar error
            
        new_names = []
        for col in names:
            # Agregamos las nuevas columnas que genera el transform
            new_names.append(f"{col}_sin")
            new_names.append(f"{col}_cos")
            # Si NO borramos la original, también la incluimos (aunque tu default es True)
            if not self.drop_original:
                new_names.append(col)
                
        return new_names


def procesar_fechas(df):
    # Es vital hacer una copia para no afectar el dataframe original fuera del pipe
    X = df.copy()
    
    # Tu lógica de conversión
    if 'Date' in X.columns:
        X['Date'] = pd.to_datetime(X['Date'])
        X['Mes'] = X['Date'].dt.month
        
        # Mapeo de Temporada (puedes traer tu diccionario aquí o tenerlo global)
        estaciones = {
            12: 'Verano', 1: 'Verano', 2: 'Verano',
            3: 'Otoño', 4: 'Otoño', 5: 'Otoño',
            6: 'Invierno', 7: 'Invierno', 8: 'Invierno',
            9: 'Primavera', 10: 'Primavera', 11: 'Primavera'
        }
        X['Temporada'] = X['Mes'].apply(lambda x: estaciones.get(x))
        
        # Opcional: Eliminar la columna Date si ya no sirve
        # X = X.drop(columns=['Date'])
        
    return X

# Creamos el transformador para el pipeline
transformer_fechas = FunctionTransformer(procesar_fechas, validate=False)