from matplotlib import pyplot as plt
import pickle

f = open("data.list", "r+b")
p = pickle.load(f)
f.close()

for img in p[0]:
  plt.imshow(img)
  plt.show()


