from embed import PoincareEmbedding
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cProfile
import time
import os
from datetime import datetime
from gensim.models.poincare import PoincareModel

# Data used was from https://github.com/facebookresearch/poincare-embeddings/blob/master/wordnet/transitive_closure.py

mammals = pd.read_csv('data/mammals_facebook.csv')
closure = list(zip(mammals['id1'], mammals['id2']))

pr = cProfile.Profile()
pr.enable()

embedding = PoincareEmbedding(closure, dimensions=2, num_negs=100, lr=0.1, burn_in=True, batch_size=1000, print=True, gpu=False)
embedding.fit_transform(300)

pr.disable()
pr.print_stats(sort='time')

# time.sleep(1)
# points = list(embedding.embedding)
#
# plotly_fig = go.Figure(data=go.Scattergl(x=[x[0] for x in points], y=[x[1] for x in points], mode='markers', text=embedding.vocab))
# plotly_fig.show()
#
# pr = cProfile.Profile()
# pr.enable()
# print("starting")
#
# model = PoincareModel(closure, negative=100)
# model.train(epochs=300, batch_size=1000)
#
# points = model.kv.syn0
# xs=[x[0] for x in points]
# ys=[x[1] for x in points]
#
# plotly_fig = go.Figure(data=go.Scattergl(x=xs, y=ys, mode='markers', text=model.kv.index2word))
# plotly_fig.show()
#
# pr.disable()
# pr.print_stats(sort='time')