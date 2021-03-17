from dataset_chap123 import *
from mlp_model import *


ad = AbaloneDataset()
am = MlpModel('abalone_model', ad, [])
am.exec_all(epoch_count=10, report=2)


pd = PulsarDataset()
pm = MlpModel('pulsar_model', pd, [4])
pm.exec_all()

