import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
from sklearn.decomposition import PCA

def load_timeseries_data_from_folder(folder, label):
    sequences = []
    filenames = []
    for filename in os.listdir(folder):
        if filename.endswith(".csv"):
            try:
                file_path = os.path.join(folder, filename)
                data = pd.read_csv(file_path, header=None).values
                if data.shape == (30, 8):
                    sequences.append(data.flatten())
                    filenames.append("{}_{}".format(label, filename))
                else:
                    print("ファイル {}: 予期しない形状 {}".format(filename, data.shape))
            except Exception as e:
                print("ファイル {} を読み込み中にエラー: {}".format(filename, e))
    return np.array(sequences), filenames

person_a_folder = ''
person_b_folder = ''
person_c_folder = ''

PATA = ''
PATB = ''
PATC = ''

data_a, filenames_a = load_timeseries_data_from_folder(person_a_folder, PATA)
data_b, filenames_b = load_timeseries_data_from_folder(person_b_folder, PATB)
data_c, filenames_c = load_timeseries_data_from_folder(person_c_folder, PATC)

def plot_pca_3d(data_a, data_b, data_c, filenames_a, filenames_b, filenames_c):
    data = np.vstack((data_a, data_b, data_c))
    filenames = filenames_a + filenames_b + filenames_c
    labels = np.array([0] * len(data_a) + [1] * len(data_b) + [2] * len(data_c))
    
    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(data)
    
    explained_variance = pca.explained_variance_ratio_
    print("寄与率: PC1={:.2f}, PC2={:.2f}, PC3={:.2f}".format(*explained_variance))

    reduced_data[:, 0] = -reduced_data[:, 0]
    
    df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2', 'PC3'])
    df['label'] = labels
    df['label'] = df['label'].map({0: PATA, 1: PATB, 2: PATC})
    df['filename'] = filenames
    
    marker_shapes = df['label'].map({PATA: 'circle', PATB: 'circle', PATC: 'circle'})
    marker_sizes = df['label'].map({PATA: 5, PATB: 5, PATC: 5})
    
    fig = px.scatter_3d(df, x='PC1', y='PC2', z='PC3', color='label', hover_name='filename',
                        symbol=marker_shapes, size=marker_sizes)
    fig.update_layout(
        title='TITLE',
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3"
        )
    )
    pio.write_html(fig, file='TITLE'+'.html', auto_open=True)

plot_pca_3d(data_a, data_b, data_c, filenames_a, filenames_b, filenames_c)
