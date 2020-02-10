import al
import sklearn.datasets
X, y = sklearn.datasets.make_classification()

m = al.AL()
m.fit(X, y)
progs = m.get_programs()
print(progs[0].pipeline_code())