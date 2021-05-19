import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np


test = pd.read_csv('../../WEIS/weis_data_decoded.csv')
test2 = test.drop(['text', 'text_ner'], axis=1)
# print(test2.shape)
obj_df = test2.select_dtypes(include=['object']).copy()
obj_df = obj_df.astype('category')
for name in list(obj_df.columns):
    test2[name] = obj_df[name].cat.codes
fig, ax = plt.subplots()
im = ax.imshow(test2.corr(method='pearson'))
ax.xaxis.tick_top()
ax.set_xticks(range(test2.shape[1]))
ax.set_yticks(range(test2.shape[1]))
ax.set_xticklabels(list(test2.columns))
ax.set_yticklabels(list(test2.columns))
plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
         rotation_mode="anchor")
fig.tight_layout()
plt.savefig('correlation2.png', bbox_inches='tight')
plt.show()
# print(list(test.columns))
# print(test2.shape)
