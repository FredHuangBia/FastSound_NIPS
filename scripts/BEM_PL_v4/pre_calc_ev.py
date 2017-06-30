'''
    Precalculation: EV part
    Usage: python pre_calc_ev.py objId matId overwrite host
'''

import sys, getopt, ConfigParser, os, json, subprocess, math

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/BEMs/'
SHAPE_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4s/shapes/'

class Obj:
    ROOT = ROOT_DIR
    def __init__(self,objId=0,matId=0):
        self.objId = objId
        self.matId = matId
        self.Load()
    def ReadMaterial(self,matId):
        self.matId = matId
        matCfg = ConfigParser.ConfigParser()
        cfgPath = os.path.join(self.ROOT,'../materials','material-%d.cfg'%(self.matId)) # TODO ln -s materials
        matCfg.read(cfgPath)

        self.materialName = matCfg.get('DEFAULT','name')
        self.youngsModulus = matCfg.getfloat('DEFAULT','youngs')
        self.poissonRatio = matCfg.getfloat('DEFAULT','poison')
        self.density = matCfg.getfloat('DEFAULT','density')
        self.alpha = matCfg.getfloat('DEFAULT','alpha')
        self.beta = matCfg.getfloat('DEFAULT','beta')
        self.friction = matCfg.getfloat('DEFAULT','friction')
        self.restitution = matCfg.getfloat('DEFAULT','restitution')
        self.rollingFriction = matCfg.getfloat('DEFAULT','rollingFriction')
        self.spinningFriction = matCfg.getfloat('DEFAULT','spinningFriction')    
    
    def ReadTet(self):
        self.tetPath = os.path.join(self.ROOT,'%02d%d' %(self.objId, self.matId),'%02d%d.tet'%(self.objId, self.matId))
        
    def Load(self):
        self.ReadMaterial(self.matId)
        self.ReadTet()
        
class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def CreateDir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def copy_shape(objId, matId, suffix, outPath):
    cmd = 'cp %s %s' %(os.path.join(SHAPE_DIR, str(objId), str(objId)+suffix), os.path.join(outPath, '%02d%d'%(objId, matId)+suffix))
    subprocess.call(cmd, shell=True)


if __name__ == '__main__':
    argv = sys.argv
    SOURCECODE = '/data/vision/billf/object-properties/sound'
    MODALSOUND = SOURCECODE + '/sound/primitives/code/ModalSound/build/bin'
    EXTMAT = MODALSOUND + '/extmat'
    GENMOMENTS = MODALSOUND +'/gen_moments'
    FILEGENERATORS = SOURCECODE + '/sound/code/file_generators'
    print 'CALL!'

    obj_id = int(argv[1])
    mat_id = int(argv[2])
    overwrite = int(argv[3])
    host = argv[4]
    print (overwrite)
    obj = Obj(obj_id,mat_id)
    outPath = os.path.join(ROOT_DIR,'%02d%d' %(obj.objId, obj.matId))
    CreateDir(outPath)
    copy_shape(obj.objId, obj.matId, '.orig.obj', outPath)
    copy_shape(obj.objId, obj.matId, '.ply', outPath)
    copy_shape(obj.objId, obj.matId, '.tet', outPath)

    renew = 'kinit -R'
    CreateDir(os.path.join(SOURCECODE,'www','EV_prim_status',host))
    logfile = os.path.join(SOURCECODE,'www','EV_prim_status',host,'%02d%d.txt' %(obj.objId,obj.matId))
    subprocess.call('echo start > %s' %logfile,shell=True)
    #call extmat, save to root
    with cd(outPath):
        # call extmat            
        if not os.path.exists('%02d%d.stiff.spm' %(obj.objId, obj.matId)) or not os.path.exists('%02d%d.mass.spm' %(obj.objId, obj.matId)) or overwrite == 1:
            cmd = EXTMAT + ' -f %02d%d -y %.4g -p %.5g -m -k -g -s -d 1 | tee -a %s' %(obj.objId, obj.matId,obj.youngsModulus,obj.poissonRatio,logfile);
            subprocess.call(cmd ,shell=True)
            
        #call ev calculation
        print('EV!')
        if not os.path.exists('%02d%d.ev' %(obj.objId, obj.matId)) or overwrite == 1:
            subprocess.call(renew ,shell=True)
            cmd = 'matlab -nodisplay -nodesktop -nosplash -r "addpath(\'%s\'); ev_generator60(\'%02d%d\', 60); quit"| tee -a %s' %(FILEGENERATORS,obj.objId, obj.matId,logfile)
            subprocess.call(cmd ,shell=True)
    
        #Geo maps
        if not os.path.exists('%02d%d.vmap' %(obj.objId, obj.matId)) or overwrite == 1:
            cmd = '%s/vmap_generator %02d%d.geo.txt %02d%d.vmap | tee -a %s' %(FILEGENERATORS,obj.objId, obj.matId,obj.objId, obj.matId,logfile)
            print cmd
            subprocess.call(cmd ,shell=True)

        #BEM input
        CreateDir(os.path.join(outPath,'bem_input'))
        CreateDir(os.path.join(outPath,'bem_result'))
        CreateDir(os.path.join(outPath,'fastbem'))
        if not os.path.exists('./bem_input/init_bem.mat') or \
            not os.path.exists('./bem_input/mesh.mat') or \
            not os.path.exists('./bem_input/init_bem.mat') or overwrite == 1:
            subprocess.call(renew ,shell=True)
            cmd = 'matlab -nodisplay -nodesktop -nosplash -r "addpath(\'%s\');BEMInputGenerator(\'%s\', \'%02d%d\', %.5g, %.5g, %.5g,%d); quit" | tee -a %s' %(FILEGENERATORS,outPath,obj.objId, obj.matId,obj.density,obj.alpha,obj.beta,overwrite,logfile)
            subprocess.call(cmd ,shell=True)
