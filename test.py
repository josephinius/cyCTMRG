import cytnx


uT = cytnx.UniTensor(cytnx.ones([2,3,4]), name="untagged tensor", labels=["a","b","c"])
uT.print_diagram()

