import pandas as pd

bc_dataset = pd.read_csv("Breas Cancer.csv", header=0).values

print(bc_dataset[0:5,2:32])