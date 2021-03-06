#!/bin/bash

check_exists() {
    if [ ! -e $1 ]; then
        echo "$1 does not exist" >&2
        exit 1
    fi
}

check_file() {
    echo $1
    check_exists $1
    if [ ! -f $1 ]; then
        echo "$1 is not a file" >&2
        exit 1
    fi
}

check_directory() {
    check_exists $1
    if [ ! -d $1 ]; then
        echo "$1 is not a directory" >&2
        exit 1
    fi
}

SNAPSHOT_DIR="snapshots"
DEPLOY_FILE="deploy.prototxt"
while getopts ":hs:d:c:i:" opt; do
    case $opt in
        h)
            # TODO: Improve help message
            echo "Here's some help"
            exit 0
            ;;
        s)
            check_directory $OPTARG
            SNAPSHOT_DIR=$OPTARG
            ;;
        d)
            check_file $OPTARG
            DEPLOY_FILE=$OPTARG
            ;;
        i)
            check_directory $OPTARG
            IMAGES_DIR=$OPTARG
            ;;
        :)
            echo "Option -$OPTARG requires an argument" >&2
            exit 1
            ;;
        \?)
            echo "Invalid option: -$OPTARG" >&2
            exit 1
    esac
done
# Pop the arguments parsed by getopts off the argument stack
shift $((OPTIND - 1))

# Start running analysis
for taskdir in $@; do
	task=$(basename $taskdir)
	echo "Collecting statistics for $task"
	mkdir -p out/$task
	snapshot=$(ls -t $taskdir/snapshots/*.caffemodel | head -1)

	python visualize_weights.py --save out/$task/weights-conv1.jpg $taskdir/snapshots/ $taskdir/deploy.prototxt conv1 2> stderr.log

	python visualize_weights.py --save out/$task/weights-conv2.jpg $taskdir/snapshots/ $taskdir/deploy.prototxt conv2 2> stderr.log

	python visualize_net.py --save out/$task/filters.jpg $snapshot $taskdir/deploy.prototxt kernel > stdout.log 2> stderr.log

    if [ -n "$IMAGES_DIR" ]; then
        python visualize_net.py --save out/$task/output.jpg --shuffle --images $IMAGES_DIR $snapshot $taskdir/deploy.prototxt output > stdout.log 2> stderr.log
    fi
done
