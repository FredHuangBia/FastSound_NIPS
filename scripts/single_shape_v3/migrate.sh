#!/bin/sh

INDIR=/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3
OUTDIR=/data/vision/billf/object-properties/sound/sound/primitives/data/v2b

PRIM=$1
SUB=$2
MAT0=$3
MAT1=$4
NEWID=$5

if [ ! -d ${OUTDIR}/${NEWID} ]; then
  mkdir ${OUTDIR}/${NEWID}
fi

cd ${OUTDIR}/${NEWID}
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/${PRIM}-${SUB}.ev ./${NEWID}.ev
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/${PRIM}-${SUB}.geo.txt ./${NEWID}.geo.txt
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/${PRIM}-${SUB}.vmap ./${NEWID}.vmap
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/${PRIM}-${SUB}.mass.spm ./${NEWID}.mass.spm
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/${PRIM}-${SUB}.stiff.spm ./${NEWID}.stiff.spm
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/${PRIM}-${SUB}.obj ./${NEWID}.obj
cp ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/log.txt ./log.txt
cp ${INDIR}/${PRIM}/${SUB}/${PRIM}-${SUB}.tet ./${NEWID}.tet
cp ${INDIR}/${PRIM}/${SUB}/${PRIM}-${SUB}.orig.obj ./${NEWID}.orig.obj
cp ${INDIR}/${PRIM}/${SUB}/volume.txt ./volume.txt

ln -s ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/bem_input ./
ln -s ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/bem_output ./
ln -s ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/fastbem ./
ln -s ${INDIR}/${PRIM}/${SUB}/mat-${MAT0}-${MAT1}/moments ./