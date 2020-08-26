import os
import pandas as pd
from sklearn.model_selection import train_test_split

save_ratio = 0.5
x = pd.read_csv('data/libri_train_manifest.csv', header=None)
to_save, to_discard = train_test_split(x, test_size = save_ratio)
to_save.to_csv('data/libri'+str(save_ratio)+'_train_manifest.csv', index = False, header = False)

help(train_test_split)
