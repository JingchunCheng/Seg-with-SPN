import os,sys
sys.path.insert(0, "../fcn_python/")
sys.path.insert(0, "../python_layers/")


import caffe
import surgery

import numpy as np


weights      = sys.argv[1]
solver_proto = sys.argv[2]

caffe.set_mode_gpu()

solver = caffe.SGDSolver(solver_proto)
solver.net.copy_from(weights)

# surgeries
interp_layers = [k for k in solver.net.params.keys() if 'up' in k]
surgery.interp(solver.net, interp_layers)

for _ in range(1):
    solver.step(31000)


