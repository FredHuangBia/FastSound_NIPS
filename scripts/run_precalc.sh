SOURCECODE=/data/vision/billf/object-properties/sound
SOURCEPATH=/data/vision/billf/object-properties/sound/sound/primitives/scripts
LIBPATH=/data/vision/billf/object-properties/sound/software/fmmlib3d-1.2/matlab

bash ~/.renew
#echo $1
source /data/vision/billf/object-properties/sound/ztzhang/bash_profile
HOSTNAME=hostname
python /data/vision/billf/object-properties/sound/sound/primitives/scripts/pre_calc_ev.py $1 $2 0 $HOSTNAME
echo "finished calling"
#echo $2
cd $SOURCEPATH
bash ~/.renew
source /data/vision/billf/object-properties/sound/ztzhang/bash_profile
MATLAB=/afs/csail.mit.edu/common/matlab/2015a/bin/matlab
CURPATH=/data/vision/billf/object-properties/sound/sound/primitives/data/v1/$1/mat-$2
$MATLAB -nodisplay -nodesktop -nosplash -r "addpath('${SOURCEPATH}');addpath('${LIBPATH}'); FMMsolver('$CURPATH',0); quit"

cd $CURPATH
mkdir -p moments
cd moments
if [ -f "moments.pbuf" ]
then
    echo "FOUND!!!"
else
    /data/vision/billf/object-properties/sound/sound/code/ModalSound/build/bin/gen_moments ../fastbem/input-%d.dat ../bem_result/output-%d.dat 0 29
fi
