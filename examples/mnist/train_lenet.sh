#!/usr/bin/env sh
set -e

caffe train --solver=examples/mnist/lenet_solver.prototxt $@
