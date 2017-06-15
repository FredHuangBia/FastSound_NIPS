import sys, getopt, ConfigParser, os, json, subprocess, math

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/'

class Obj:
    ROOT = ROOT_DIR
    def __init__(self,objId=(0,0),matId=(0,0)):
        self.objId = objId
        self.matId = matId
        self.Load()
    def ReadMaterial(self,matId):
        self.matId = matId
        matCfg = ConfigParser.ConfigParser()
        cfgPath = os.path.join(self.ROOT,'materials','material-%d-%d.cfg'%(self.matId[0],self.matId[1])) # TODO ln -s materials
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
        self.tetPath = os.path.join(self.ROOT,'%s'%self.objId[0],'%s'%self.objId[1],'%s-%s.tet'%(self.objId[0],self.objId[1]))
        
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
        
if __name__ == '__main__':
    argv = sys.argv
    SOURCECODE = '/data/vision/billf/object-properties/sound'
    MODALSOUND = SOURCECODE + '/sound/code/ModalSound/build/bin'
    EXTMAT = MODALSOUND + '/extmat'
    GENMOMENTS = MODALSOUND +'/gen_moments'
    FILEGENERATORS = SOURCECODE + '/sound/code/file_generators'
    print('CALL!')

    obj_id = (argv[1],argv[2])
    mat_id = (int(argv[3]), int(argv[4]))
    overwrite = int(argv[5])
    host = argv[6]
    print (overwrite)
    obj = Obj(obj_id,mat_id)
    outPath = os.path.join(ROOT_DIR,'%s/%s' %(obj.objId[0],obj.objId[1]),'mat-%d-%d' %(obj.matId[0],obj.matId[1]))
    CreateDir(outPath)
    renew = 'kinit -R'
    CreateDir(os.path.join(SOURCECODE,'www','EV_prim_status',host))
    logfile = os.path.join(SOURCECODE,'www','EV_prim_status',host,'%s-%s-%d-%d.txt' %(obj.objId[0],obj.objId[1],obj.matId[0],obj.matId[1]))
    subprocess.call('echo start > %s' %logfile,shell=True)
    #call extmat, save to root
    with cd(outPath):
        if not os.path.exists('%s-%s.tet' %(obj.objId[0],obj.objId[1])) or overwrite == 1 :
            subprocess.call('ln -s ../%s-%s.tet %s-%s.tet' %(obj.objId[0],obj.objId[1],obj.objId[0],obj.objId[1]),shell=True)
            
        if not os.path.exists('%s-%s.stiff.spm' %(obj.objId[0],obj.objId[1])) or not os.path.exists('%s-%s.mass.spm' %(obj.objId[0],obj.objId[1])) or overwrite == 1:
            cmd = EXTMAT + ' -f %s-%s -y %.4g -p %.5g -m -k -g -s -d 1 | tee -a %s' %(obj.objId[0],obj.objId[1],obj.youngsModulus,obj.poissonRatio,logfile);
            subprocess.call(cmd ,shell=True)
            
        #call ev calculation, save to mat-id
        print('EV!')
        if not os.path.exists('%s-%s.ev' %(obj.objId[0],obj.objId[1])) or overwrite == 1:
            subprocess.call(renew ,shell=True)
            cmd = 'matlab -nodisplay -nodesktop -nosplash -r "addpath(\'%s\'); ev_generator60(\'%s-%s\', 60); quit"| tee -a %s' %(FILEGENERATORS,obj.objId[0],obj.objId[1],logfile)
            subprocess.call(cmd ,shell=True)
    
        #Geo maps
        if not os.path.exists('%s-%s.vmap' %(obj.objId[0],obj.objId[1])) or overwrite == 1:
            cmd = '%s/vmap_generator %s-%s.geo.txt %s-%s.vmap | tee -a %s' %(FILEGENERATORS,obj.objId[0],obj.objId[1],obj.objId[0],obj.objId[1],logfile)
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
            cmd = 'matlab -nodisplay -nodesktop -nosplash -r "addpath(\'%s\');BEMInputGenerator(\'%s\', \'%s-%s\', %.5g, %.5g, %.5g,%d); quit" | tee -a %s' %(FILEGENERATORS,outPath,obj.objId[0],obj.objId[1],obj.density,obj.alpha,obj.beta,overwrite,logfile)
            subprocess.call(cmd ,shell=True)
