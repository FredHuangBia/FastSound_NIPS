# script calling ev, bem and gen_moments

SOURCEPATH=/data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL
LIBPATH=/data/vision/billf/object-properties/sound/software/fmmlib3d-1.2/matlab
CURPATH=/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/$1/$2/mat-$3-$4

kinit -R
source /data/vision/billf/object-properties/sound/ztzhang/bash_profile

echo "CALLING EV" # TODO for those do not need recalculate ev
kinit -R
python /data/vision/billf/object-properties/sound/sound/primitives/scripts/BEM_PL/pre_calc_ev.py $1 $2 $3 $4 0 $HOSTNAME
echo "finished calling EV"

cd $SOURCEPATH
source /data/vision/billf/object-properties/sound/ztzhang/bash_profile
echo "CALLING MATLAB"
kinit -R
matlab -nodisplay -nodesktop -nosplash -r "addpath('${SOURCEPATH}');BEMsolver('$CURPATH',0); quit"
echo "finished calling MATLAB"

cd $CURPATH
mkdir -p moments
cd moments
if [ -f "moments.pbuf" ]
then
    echo "MOMENTS FOUND!!!"
else
	echo "CALLING GEN_MOMENTS"
    /data/vision/billf/object-properties/sound/sound/code/ModalSound/build/bin/gen_moments ../fastbem/input-%d.dat ../bem_result/output-%d.dat 0 59
    echo "finished calling GEN_MOMENTS"
fi
