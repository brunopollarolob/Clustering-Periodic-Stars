#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para visualización de datos y resultados del análisis.
Permite visualizar curvas de luz, distribuciones de características y anomalías detectadas.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import lightkurve as lk
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Añadir el directorio raíz al path para importar módulos locales
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clustering_stars.utils.visualization import visualize_latent_space, plot_feature_distributions_by_class

def visualize_class_distribution(df, class_column='alerceclass', output_file=None):
    """
    Visualiza la distribución de clases en los datos.
    
    Args:
        df (pd.DataFrame): DataFrame con datos y etiquetas.
        class_column (str): Nombre de la columna de clases.
        output_file (str): Ruta para guardar el gráfico (opcional).
    """
    plt.figure(figsize=(10, 6))
    counts = df[class_column].value_counts()
    ax = counts.plot.bar(color=sns.color_palette("husl", len(counts)))
    
    # Añadir conteos en las barras
    for i, count in enumerate(counts):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    plt.title(f'Distribución de clases ({len(df)} objetos)')
    plt.xlabel('Clase')
    plt.ylabel('Número de objetos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file)
        print(f"Gráfico guardado en {output_file}")
    
    plt.show()
    
    # Imprimir tabla de distribución
    print("\nDistribución de clases:")
    print(counts)
    print(f"\nTotal de objetos: {len(df)}")

def visualize_features(df, class_column='alerceclass', n_features=6, output_dir=None):
    """
    Visualiza las características más importantes por clase.
    
    Args:
        df (pd.DataFrame): DataFrame con características y etiquetas.
        class_column (str): Nombre de la columna de clases.
        n_features (int): Número de características a visualizar.
        output_dir (str): Directorio para guardar gráficos (opcional).
    """
    # Seleccionar todas las columnas excepto la columna de clase
    feature_columns = df.columns.drop(class_column).tolist()
    
    # Seleccionar las primeras n características
    feature_columns = feature_columns[:n_features]
    
    print(f"Visualizando las primeras {len(feature_columns)} características...")
    
    plot_feature_distributions_by_class(df, class_column, feature_columns)
    
    # Si se especificó un directorio de salida, guardar cada característica por separado
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Crear paleta de colores para las clases
        palette = sns.color_palette("husl", len(df[class_column].unique()))
        
        for feature in feature_columns:
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=class_column, y=feature, data=df, palette=palette)
            plt.title(f'Distribución de {feature} por clase')
            plt.xlabel('Clase')
            plt.ylabel(feature)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            output_file = os.path.join(output_dir, f'feature_{feature}.png')
            plt.savefig(output_file)
            plt.close()
            
        print(f"Gráficos guardados en {output_dir}")

def visualize_light_curves(lcs_file, oids_file, top_n=5, period_column='Multiband_period_g_r', output_dir=None):
    """
    Visualiza las curvas de luz para los objetos de cada clase.
    
    Args:
        lcs_file (str): Ruta al archivo de curvas de luz (.parquet).
        oids_file (str): Ruta al archivo de objetos (.parquet).
        top_n (int): Número de objetos a visualizar por clase.
        period_column (str): Nombre de la columna con el periodo.
        output_dir (str): Directorio para guardar gráficos (opcional).
    """
    print(f"Cargando curvas de luz desde {lcs_file} y objetos desde {oids_file}...")
    lcs = pd.read_parquet(lcs_file)
    oids = pd.read_parquet(oids_file)
    
    # Agrupar por clase
    class_column = 'alerceclass' if 'alerceclass' in oids.columns else oids.columns[1]
    grouped = oids.groupby(class_column)
    
    # Crear directorios si se especificó
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar curvas de luz para cada clase
    for class_name, group in grouped:
        print(f"Visualizando curvas de luz para la clase {class_name}...")
        
        # Seleccionar top_n objetos
        top_oids = group.head(top_n).index.tolist()
        
        # Configurar subplot
        fig, axes = plt.subplots(top_n, 2, figsize=(15, 4 * top_n))
        
        # Visualizar cada objeto
        for i, oid in enumerate(top_oids):
            # Filtrar datos para el OID actual
            subset = lcs[lcs['oid'] == oid]
            
            # Verificar si hay datos para este OID
            if len(subset) == 0:
                axes[i, 0].text(0.5, 0.5, f"No hay datos disponibles para OID: {oid}", ha='center')
                axes[i, 1].text(0.5, 0.5, f"No hay datos disponibles para OID: {oid}", ha='center')
                continue
            
            # Crear un objeto LightCurve
            time = subset['mjd'].values
            flux = subset['magpsf'].values
            flux_err = subset['sigmapsf'].values if 'sigmapsf' in subset.columns else None
            
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
            
            # Obtener el periodo si está disponible
            try:
                period = oids.loc[oid, period_column]
                lc_folded = lc.fold(period=period)
                title_suffix = f" - Period: {period:.4f}"
            except:
                lc_folded = lc
                title_suffix = " - No period available"
            
            # Curva de luz original
            axes[i, 0].errorbar(lc.time, lc.flux, yerr=lc.flux_err, fmt='o', markersize=5)
            axes[i, 0].set_title(f"Light Curve - OID: {oid} - Class: {class_name}")
            axes[i, 0].set_xlabel("Time (MJD)")
            axes[i, 0].set_ylabel("Magnitude")
            axes[i, 0].invert_yaxis()  # Las magnitudes son invertidas
            
            # Curva de luz plegada (folded)
            axes[i, 1].errorbar(lc_folded.time, lc_folded.flux, yerr=lc_folded.flux_err, fmt='o', markersize=5)
            axes[i, 1].set_title(f"Folded Light Curve - OID: {oid}{title_suffix}")
            axes[i, 1].set_xlabel("Phase")
            axes[i, 1].set_ylabel("Magnitude")
            axes[i, 1].invert_yaxis()
        
        plt.tight_layout()
        
        # Guardar si se especificó un directorio
        if output_dir:
            output_file = os.path.join(output_dir, f"light_curves_{class_name}.png")
            plt.savefig(output_file)
            print(f"Curvas de luz para la clase {class_name} guardadas en {output_file}")
        
        plt.show()

def visualize_anomalies(results_file, lcs_file, oids_file, output_dir=None):
    """
    Visualiza las anomalías detectadas.
    
    Args:
        results_file (str): Ruta al archivo con resultados de predicciones.
        lcs_file (str): Ruta al archivo de curvas de luz (.parquet).
        oids_file (str): Ruta al archivo de objetos (.parquet).
        output_dir (str): Directorio para guardar gráficos (opcional).
    """
    print("Cargando resultados y datos...")
    results = pd.read_csv(results_file)
    lcs = pd.read_parquet(lcs_file)
    oids = pd.read_parquet(oids_file)
    
    # Filtrar solo anomalías
    anomalies = results[results['is_anomaly'] == 1]
    
    print(f"Visualizando {len(anomalies)} anomalías detectadas...")
    
    # Agrupar anomalías por clase
    class_column = 'alerceclass' if 'alerceclass' in results.columns else 'class_name'
    grouped = anomalies.groupby(class_column)
    
    # Crear directorios si se especificó
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualizar anomalías para cada clase
    for class_name, group in grouped:
        print(f"Visualizando anomalías para la clase {class_name}...")
        
        # Limitar a top 5 anomalías para no saturar
        top_anomalies = group.head(5)
        
        # Configurar subplot
        fig, axes = plt.subplots(len(top_anomalies), 2, figsize=(15, 4 * len(top_anomalies)))
        
        # Si solo hay una anomalía, convertir axes a array 2D
        if len(top_anomalies) == 1:
            axes = np.array([axes])
        
        # Visualizar cada anomalía
        for i, (_, row) in enumerate(top_anomalies.iterrows()):
            oid = row['oid']
            
            # Filtrar datos para el OID actual
            subset = lcs[lcs['oid'] == oid]
            
            if len(subset) == 0:
                axes[i, 0].text(0.5, 0.5, f"No hay datos disponibles para OID: {oid}", ha='center')
                axes[i, 1].text(0.5, 0.5, f"No hay datos disponibles para OID: {oid}", ha='center')
                continue
            
            # Crear un objeto LightCurve
            time = subset['mjd'].values
            flux = subset['magpsf'].values
            flux_err = subset['sigmapsf'].values if 'sigmapsf' in subset.columns else None
            
            lc = lk.LightCurve(time=time, flux=flux, flux_err=flux_err)
            
            # Obtener el periodo si está disponible
            try:
                period = oids.loc[oid, 'Multiband_period_g_r']
                lc_folded = lc.fold(period=period)
                title_suffix = f" - Period: {period:.4f}"
            except:
                lc_folded = lc
                title_suffix = " - No period available"
                
            # Obtener score de anomalía si está disponible
            anomaly_score = f" - Score: {row['anomaly_score']:.3f}" if 'anomaly_score' in row else ""
            
            # Curva de luz original
            axes[i, 0].errorbar(lc.time, lc.flux, yerr=lc.flux_err, fmt='o', markersize=5)
            axes[i, 0].set_title(f"Anomalía - OID: {oid} - Clase: {class_name}{anomaly_score}")
            axes[i, 0].set_xlabel("Tiempo (MJD)")
            axes[i, 0].set_ylabel("Magnitud")
            axes[i, 0].invert_yaxis()
            
            # Curva de luz plegada (folded)
            axes[i, 1].errorbar(lc_folded.time, lc_folded.flux, yerr=lc_folded.flux_err, fmt='o', markersize=5)
            axes[i, 1].set_title(f"Anomalía (Folded) - OID: {oid}{title_suffix}")
            axes[i, 1].set_xlabel("Fase")
            axes[i, 1].set_ylabel("Magnitud")
            axes[i, 1].invert_yaxis()
        
        plt.tight_layout()
        
        # Guardar si se especificó un directorio
        if output_dir:
            output_file = os.path.join(output_dir, f"anomalies_{class_name}.png")
            plt.savefig(output_file)
            print(f"Anomalías para la clase {class_name} guardadas en {output_file}")
        
        plt.show()

def main():
    parser = argparse.ArgumentParser(
        description='Visualización de datos y resultados del análisis')
    
    parser.add_argument('--action', type=str, required=True, 
                    choices=['classes', 'features', 'lightcurves', 'anomalies'],
                    help='Acción a realizar: visualizar distribución de clases, características, curvas de luz o anomalías')
    parser.add_argument('--data_file', type=str, default=None,
                    help='Ruta al archivo de características (.parquet)')
    parser.add_argument('--labels_file', type=str, default=None,
                    help='Ruta al archivo de etiquetas (.parquet)')
    parser.add_argument('--lcs_file', type=str, default=None,
                    help='Ruta al archivo de curvas de luz (.parquet)')
    parser.add_argument('--results_file', type=str, default=None,
                    help='Ruta al archivo con resultados de predicciones (.csv)')
    parser.add_argument('--output_dir', type=str, default=None,
                    help='Directorio para guardar gráficos')
    parser.add_argument('--n_features', type=int, default=6,
                    help='Número de características a visualizar')
    parser.add_argument('--top_n', type=int, default=5,
                    help='Número de objetos a visualizar por clase')
    
    args = parser.parse_args()
    
    # Ejecutar la acción correspondiente
    if args.action == 'classes':
        if not args.data_file or not args.labels_file:
            parser.error("Para visualizar distribución de clases, se requieren --data_file y --labels_file")
        
        # Cargar datos y etiquetas
        features = pd.read_parquet(args.data_file)
        labels = pd.read_parquet(args.labels_file)
        
        # Preparar DataFrame
        if 'oid' in features.columns:
            features = features.set_index('oid')
        if 'oid' in labels.columns:
            labels = labels.set_index('oid')
        
        # Concatenar etiquetas y características
        class_column = 'alerceclass' if 'alerceclass' in labels.columns else labels.columns[0]
        df_concatenated = pd.concat([labels, features], axis=1)
        
        # Visualizar distribución de clases
        output_file = os.path.join(args.output_dir, 'class_distribution.png') if args.output_dir else None
        visualize_class_distribution(df_concatenated, class_column, output_file)
    
    elif args.action == 'features':
        if not args.data_file or not args.labels_file:
            parser.error("Para visualizar características, se requieren --data_file y --labels_file")
        
        # Cargar datos y etiquetas
        features = pd.read_parquet(args.data_file)
        labels = pd.read_parquet(args.labels_file)
        
        # Preparar DataFrame
        if 'oid' in features.columns:
            features = features.set_index('oid')
        if 'oid' in labels.columns:
            labels = labels.set_index('oid')
        
        # Concatenar etiquetas y características
        class_column = 'alerceclass' if 'alerceclass' in labels.columns else labels.columns[0]
        df_concatenated = pd.concat([labels, features], axis=1)
        
        # Visualizar características
        visualize_features(df_concatenated, class_column, args.n_features, args.output_dir)
    
    elif args.action == 'lightcurves':
        if not args.lcs_file or not args.labels_file:
            parser.error("Para visualizar curvas de luz, se requieren --lcs_file y --labels_file")
        
        # Visualizar curvas de luz
        visualize_light_curves(args.lcs_file, args.labels_file, args.top_n, output_dir=args.output_dir)
    
    elif args.action == 'anomalies':
        if not args.results_file or not args.lcs_file or not args.labels_file:
            parser.error("Para visualizar anomalías, se requieren --results_file, --lcs_file y --labels_file")
        
        # Visualizar anomalías
        visualize_anomalies(args.results_file, args.lcs_file, args.labels_file, args.output_dir)
    
if __name__ == "__main__":
    main()