#!/usr/bin/env sh
# This script converts the mnist data into lmdb/leveldb format,
# depending on the value assigned to $BACKEND.
set -e

EXAMPLE=examples/multi_label
DATA=data/multi_label
BACKEND="lmdb"

echo "Creating ${BACKEND}..."

rm -rf $EXAMPLE/multi_label_train_${BACKEND}

./build/tools/convert_imageset_multi_label $DATA/ $DATA/train.txt \
  $EXAMPLE/multi_label_train_${BACKEND} --backend=${BACKEND} --shuffle=false

echo "Done."
