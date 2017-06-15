#!/bin/sh
ROOT=/data/vision/billf/object-properties/sound/sound/primitives/scripts/multi_shape_v4
DIRECTORY=$1
# echo $1 $2 $3 $4 $5 $6
if [ -d "$DIRECTORY" ]; then
	rm -r $DIRECTORY
	#echo $DIRECTORY DELETED!
fi
mkdir $DIRECTORY
cd $DIRECTORY
cp -r $ROOT/config ./
sed -i "/center/c\center = [0.0, ${2}, 0.0]" ./config/pose/pose-0-0.cfg
sed -i "/rotation/c\rotation = ${3}" ./config/pose/pose-0-0.cfg
sed -i "/alpha/c\alpha = ${4}" ./config/material/mat-0-0-0.cfg
sed -i "/beta/c\beta = ${5}" ./config/material/mat-0-0-0.cfg
sed -i "/restitution/c\restitution = ${6}" ./config/material/mat-0-0-0.cfg
sed -i "/r = 5/c\r = ${7}" ./config/camera/camera_1000.cfg

# echo NEW CONFIG GENERATED!
