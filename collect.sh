#!/bin/bash

# Make sure caffe is on $PYTHONPATH
CAFFEPY=$HOME/caffe/python
if [ -d "$CAFFEPY" ] && [[ ":$PYTHONPATH:" != *":$CAFFEPY:"* ]]; then
	export PYTHONPATH="${PYTHONPATH:+"$PYTHONPATH:"}$CAFFEPY"
fi

# Activate our virtual environment
source $HOME/venvs/caffe/bin/activate

# Start running analysis
for taskdir in $HOME/caffe-models/task*; do
	task=$(basename $taskdir)
	echo "Collecting statistics for $task"
	mkdir -p out/$task
	snapshot=$(ls $taskdir/snapshots/*.caffemodel | tail -1)
	python visualize_weights.py --save out/$task/weights-conv1.jpg $taskdir/snapshots/ $taskdir/deploy.prototxt conv1 2> stderr.log
	python visualize_weights.py --save out/$task/weights-conv2.jpg $taskdir/snapshots/ $taskdir/deploy.prototxt conv2 2> stderr.log
	python visualize_net.py --save out/$task/filters.jpg $snapshot $taskdir/deploy.prototxt >> stdout.log 2> stderr.log
done

