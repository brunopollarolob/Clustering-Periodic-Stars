import pandas as pd
from imblearn.over_sampling import SMOTE

def fill_missing_values(df, numeric_columns):
    """
    Llena valores nulos en columnas numéricas con la mediana de cada columna.

    Args:
        df (pd.DataFrame): DataFrame con datos.
        numeric_columns (list): Lista de nombres de columnas numéricas.

    Returns:
        pd.DataFrame: DataFrame con valores nulos llenados.
    """
    median_values = df[numeric_columns].median()
    return df.fillna(median_values)

def balance_data(X, y, random_state=10):
    """
    Aplica SMOTE para balancear las clases en un conjunto de datos.

    Args:
        X (pd.DataFrame or np.array): Características.
        y (pd.Series or np.array): Etiquetas.
        random_state (int): Semilla para la aleatoriedad.

    Returns:
        pd.DataFrame, pd.Series: Datos balanceados (X_resampled, y_resampled).
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

def prepare_data(oids, feats_by_oid, target_column='alerceclass', random_state=10):
    """
    Llena valores nulos, concatena etiquetas y características, y balancea los datos.

    Args:
        oids (pd.DataFrame): DataFrame con etiquetas.
        feats_by_oid (pd.DataFrame): DataFrame con características.
        target_column (str): Nombre de la columna objetivo.
        random_state (int): Semilla para SMOTE.

    Returns:
        pd.DataFrame: DataFrame con datos balanceados.
    """
    # Llenar valores nulos
    numeric_columns = feats_by_oid.select_dtypes(include=[float, int]).columns
    feats_filled = fill_missing_values(feats_by_oid, numeric_columns)

    # Concatenar etiquetas y características
    df_concatenated = pd.concat([oids[target_column], feats_filled], axis=1)
    df_concatenated.dropna(inplace=True)

    # Separar características y etiquetas
    X = df_concatenated.drop(columns=[target_column])
    y = df_concatenated[target_column]

    # Balancear datos con SMOTE
    X_resampled, y_resampled = balance_data(X, y, random_state)

    # Reconstruir DataFrame balanceado
    df_resampled = pd.concat(
        [pd.DataFrame(X_resampled), pd.DataFrame(y_resampled, columns=[target_column])],
        axis=1
    )
    return df_resampled
