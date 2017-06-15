#this is a script for Generating video with sound, using precomputed pressure field.
import sys, getopt, ConfigParser, os, json, subprocess, math
from decimal import Decimal

args = sys.argv[1:]
#argv = ['s',0,0,0,0,0,1,0]

optlist, args = getopt.getopt(args, 'bsrcp:v')
argv = [sys.argv[0]]+args
skip_bullet= False
skip_sound= False
skip_rendering= False
is_overwrite= True
skip_factor = 1
skip_video = False
for k,v in optlist:
    if k == '-b':
        skip_bullet = True
    elif k == '-s':
        skip_sound = True
    elif k == '-r':
        skip_rendering = False
    elif k =='-c':
        is_overwrite = False
    elif k=='-p':
        skip_factor = int(v)
    elif k=='-v':
        skip_video = False
print sys.argv

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v4/BEMs/'
TARGET_DIR = argv[1]

class Obj:
    ROOT = ROOT_DIR
    TARGET = TARGET_DIR
    def __init__(self,objId=0,matId=(0,0,0)):
        self.objId = objId
        self.center=[0,0,0]
        self.rotation=[0,0,0,1]
        self.matId = matId
        self.poseId = [0,0]
        self.bulletSetupId = 0
        self.mass = 1
        self.active = 1
        #self.Load()

    def ReadPose(self,poseId):
        self.poseId = poseId
        poseCfg = ConfigParser.ConfigParser()
        cfgPath = os.path.join(self.TARGET,'config','pose','pose-%d-%d.cfg'%(self.poseId[0],self.poseId[1]))
        poseCfg.read(cfgPath)
        self.center = json.loads(poseCfg.get('DEFAULT','center'))
        self.rotation = json.loads(poseCfg.get('DEFAULT','rotation'))
        self.linearVelocity = json.loads(poseCfg.get('DEFAULT','linear_velocity'))
        self.angularVelocity = json.loads(poseCfg.get('DEFAULT','angular_velocity'))
        
    def ReadBulletSetup(self,bulletSetupId):
        self.bulletSetupId = bulletSetupId
        bulletCfg = ConfigParser.ConfigParser()
        cfgPath = os.path.join(self.TARGET,'config','bullet_setup','bullet_%d.cfg'%self.bulletSetupId)
        bulletCfg.read(cfgPath)
        self.linearDamping = bulletCfg.getfloat('DEFAULT','linearDamping')
        self.angularDamping = bulletCfg.getfloat('DEFAULT','angularDamping')
        self.collisionMargin = bulletCfg.getfloat('DEFAULT','collisionMargin')

    def ReadMaterial(self,matId):
        self.matId = matId
        matCfg = ConfigParser.ConfigParser()
        cfgPath = os.path.join(self.TARGET,'config','material','mat-%d-%d-%d.cfg'%(self.matId[0],self.matId[1],self.matId[2]))
        print cfgPath
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
    
    def ReadObj(self,objId):
        self.objId = objId
        if self.active == 1:
            self.objPath = os.path.join(self.ROOT,'%d'%self.objId,'%d.orig.obj'%self.objId)
        else:
            self.objPath = os.path.join(self.ROOT,'%d'%self.objId,'%d.orig.obj'%self.objId)
        if self.active == 1:
            volumefile = open(os.path.join(self.ROOT,'%d/volume.txt'%self.objId))
            self.mass = float(volumefile.readline())*self.density/500
            print self.mass

    def Load(self):
        self.ReadPose(self.poseId)
        self.ReadBulletSetup(self.bulletSetupId)
        self.ReadMaterial(self.matId)
        self.ReadObj(self.objId)
        
    def PrintStat(self):
        print('obj #%d:\n'%self.objId)
        print('        material: #%d %d %d\n'%(self.matId[0],self.matId[1],self.matId[2]))
        print('        initial pose: #%d %d\n'%(self.poseId[0],self.poseId[1]))
        print('        bullet simulation setup: #%d\n'%self.bulletSetupId)
        
    def WriteString(self):
        properties = ''
        properties += '%.4f %.4f '%(self.mass,self.collisionMargin)
        properties += '%.2f %.2f %.2f %.2f '%tuple(self.rotation)
        properties += '%.2f %.2f %.2f 0 '%tuple(self.center)
        properties += '%.2f %.2f %.2f '%tuple(self.linearVelocity)
        properties += '%.2f %.2f %.2f '%tuple(self.angularVelocity)
        properties += '%.2f %.2f %.2f '%(self.friction, self.restitution, self.rollingFriction)
        properties += '%.2f %.2f %.2f\n'%(self.spinningFriction, self.linearDamping, self.angularDamping)
        return properties
    
    def WriteShellCmd(self):
        cmd = '-n %d -d %f -A %.3E -B %f'%(self.objId,self.density,Decimal(self.alpha),self.beta)
        return cmd

class Cam:
    ROOT = ROOT_DIR
    TARGET = TARGET_DIR
    def __init__(self,cfgId=0):
        self.cfgId = cfgId
        self.cfgPath = os.path.join(self.TARGET,'config','camera','camera_%d.cfg'%cfgId)
        self.Load()
    def SetCfgId(self,cfgId):
        self.cfgId = cfgId
        self.cfgPath = os.path.join(self.TARGET,'config','camera','camera_%d.cfg'%cfgId)
        self.Load()
    def Load(self):
        camCfg = ConfigParser.ConfigParser()
        camCfg.read(self.cfgPath)
        self.lookAt = json.loads(camCfg.get('Camera','look_at'))
        self.r = camCfg.getfloat('Camera','r')
        self.theta = camCfg.getfloat('Camera','theta')
        self.phi = camCfg.getfloat('Camera','phi')
        self.focalLength = camCfg.getint('Camera','focal_length')
        self.sensorWidth = camCfg.getint('Camera','sensor_width')
        self.CalcXYZ()
    def CalcXYZ(self):
        x = self.r*math.sin(math.radians(self.theta))*math.cos(math.radians(self.phi))
        y = self.r*math.sin(math.radians(self.theta))*math.sin(math.radians(self.phi))
        z = self.r*math.cos(math.radians(self.theta))
        self.xyz = [x,y,z]
        
class Lighting:
    ROOT = ROOT_DIR
    TARGET = TARGET_DIR
    def __init__(self,cfgId=0):
        self.cfgId = cfgId
        self.cfgPath = os.path.join(self.TARGET,'config','lighting','lighting_%d.cfg'%cfgId)
    def SetCfgId(self,cfgId):
        self.cfgId = cfgId
        self.cfgPath = os.path.join(self.TARGET,'config','lighting','lighting_%d.cfg'%cfgId)
    def PrintPath(self):
        return self.cfgPath
        
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

#################################################  MAIN  ##################################################
## This script would read in a scene configuration, which includes:
#        1.Object number 
#        2.Their initial velocity position and pose 
#        3.Camera position
#        4.Simulation Parameters, i.e. FPS, collision margin, duration, etc.
#        5.Lights
#        
#        Need to specify a scene id and corresponding obj id with materials id.
###########################################################################################################

ROOT_DIR = '/data/vision/billf/object-properties/sound/sound/primitives/data/v2b/'
BLENDER = '/data/vision/billf/object-properties/sound/software/blender/blender'
ROOT = ROOT_DIR
TARGET = TARGET_DIR

# READ SCENE CONFIG
subprocess.call('kinit -R',shell=True)
if (len(argv)<2):
    print('Not enough argument, Scene ID needed! \n')

# print argv[1]
sceneid = (int(argv[2]),int(argv[3]));
sceneConfig = ConfigParser.ConfigParser();
sceneConfig.read(os.path.join(TARGET,'config/scene/scene-%d-%d.cfg'%(sceneid[0],sceneid[1])))

# Get Objs
objnum = sceneConfig.getint('Objects','obj_num')
if(len(argv)<(4+4*objnum)):
    print 'not enough arguemnt! using default objects and materials\n'
    objs = [Obj(0,0) for x in range(objnum)]
else:
    objs = []
    for k in range(objnum):
        objs.append(Obj(objId=int(argv[k*4+4]),matId=(int(argv[k*4+5]),int(argv[k*4+6]),int(argv[k*4+7]))))
        
for objid in range(objnum):
    objs[objid].poseId[0] = (sceneConfig.getint('Objects','obj_%d_pose0'%objid))
    objs[objid].poseId[1] = (sceneConfig.getint('Objects','obj_%d_pose1'%objid))
    objs[objid].bulletSetupId = (sceneConfig.getint('Objects','obj_%d_bullet_setup'%objid))
    if sceneConfig.getint('Objects','obj_%d_is_active'%objid)==0:
        objs[objid].active = 0
        objs[objid].mass = 0
    objs[objid].Load()

#Log set up:
print('Total %d objects:\n'%objnum)

for objid in range(objnum):
    objs[objid].PrintStat()

#READ CAMERA CONFIG
cam = Cam(sceneConfig.getint('Camera','Camera'))
print('Using camera setup #%d\n'%cam.cfgId)

#READ LIGHTING CONFIG
lighting = Lighting(sceneConfig.getint('Lighting','lighting'))
print('Using lighting setup #%d\n'%lighting.cfgId)

#READ BULLET Simulation Setup
simId = 0
simCfg = ConfigParser.ConfigParser()
simCfg.read(os.path.join(TARGET,'config','simulation','sim_%d.dat'%simId))
FPS = simCfg.getint('DEFAULT','FPS')
ifGUI = simCfg.getint('DEFAULT','GUI')
duration = simCfg.getfloat('DEFAULT','duration')

# CREATE RESULT PATH
# objResultPath = 'obj'
# matResultPath = 'mat'

# for k in range(objnum):
    # objResultPath+='-%d'%objs[k].objId
#     matResultPath+='-%d-%d-%d'%(objs[k].matId[0],objs[k].matId[1],objs[k].matId[2])

resultPath = TARGET
simFilePath = os.path.join(resultPath,'sim')
renderPath = os.path.join(resultPath,'render') #
CreateDir(resultPath)
# CreateDir(renderPath) #
CreateDir(simFilePath)

# Physical simulation
# Need to prepare dat files for bullet simulation
bulletCfg = open(os.path.join(simFilePath,'bullet.cfg'), 'w')
bulletCfg.write('%d\n%d\n%.2f\n'%(ifGUI,FPS,duration))
bulletCfg.write('%.5f\n%.5f\n%.5f\n'%(cam.r,cam.theta,cam.phi))
bulletCfg.write('%.5f\n%.5f\n%.5f\n'%tuple(cam.lookAt))
bulletCfg.close()
bltInputCfg = open(os.path.join(simFilePath,'bullet.input.dat'), 'w')
bltInputCfg.write('%d\n'%(objnum))

for obj_id in range(0,objnum):
    bltInputCfg.write('%d %d '%(obj_id,objs[obj_id].objId) + objs[obj_id].WriteString())
    
bltInputCfg.close()
with cd(simFilePath):
    if skip_bullet!=True:
        for obj in objs:
            print "-------------------bullet--------%d-----------------------------"%obj.objId
            if not os.path.exists('%d.obj'%obj.objId):
                subprocess.call('ln -s %s'%os.path.join(ROOT,'%d'%obj.objId,'%d.obj'%obj.objId)\
                            +' %d.obj'%obj.objId,shell=True)
            if not os.path.exists('%d.orig.obj'%obj.objId):
                subprocess.call('ln -s %s'%os.path.join(ROOT,'%d'%obj.objId,'%d.orig.obj'%obj.objId)\
                            +' %d.orig.obj'%obj.objId,shell=True)
            #print 'ln -s %s'%os.path.join(ROOT,'data/single_shape','%d'%obj.objId,'%d.obj'%obj.objId)
            #print 'ln -s %s'%os.path.join(ROOT,'data/single_shape','%d'%obj.objId,'%d.orig.obj'%obj.objId)
        subprocess.call('sh %s '%os.path.join(ROOT,'scripts','prepare_dat.sh'),shell=True)
        print("CALLING prepare_dat.sh")
    if (skip_sound!=True):
        for obj_id in range(0,objnum):
            if(objs[obj_id].mass==0):
                continue
            #print objs[obj_id].WriteShellCmd()
            print "---------------------------%d-----------------------------"%objs[obj_id].objId
            if os.path.exists('%d.vmap'%objs[obj_id].objId):
                subprocess.call('unlink %d.vmap'%objs[obj_id].objId,shell=True)
            subprocess.call('ln -s %s'%os.path.join(ROOT,'%d'%objs[obj_id].objId,\
                            '%d.vmap'%objs[obj_id].objId)\
                            +' %d.vmap'%objs[obj_id].objId,shell=True)
           
            if os.path.exists('moments'):
                subprocess.call('unlink moments',shell=True)
            subprocess.call('ln -s %s'%os.path.join(ROOT,'%d'%objs[obj_id].objId,\
                          'moments')+' moments',shell=True)
            print 'LINK!!!'
            
            if os.path.exists('%d.geo.txt'%objs[obj_id].objId):
                subprocess.call('unlink %d.geo.txt'%objs[obj_id].objId,shell=True)  
            subprocess.call('ln -s %s'%os.path.join(ROOT,'%d'%objs[obj_id].objId,\
                          '%d.geo.txt'%objs[obj_id].objId)+\
                            ' %d.geo.txt'%objs[obj_id].objId,shell=True)
            
            if os.path.exists('%d.ev'%objs[obj_id].objId):
                subprocess.call('unlink %d.ev'%objs[obj_id].objId,shell=True)
            subprocess.call('ln -s %s'%os.path.join(ROOT,'%d'%objs[obj_id].objId,\
                              '%d.ev'%objs[obj_id].objId)+\
                            ' %d.ev'%objs[obj_id].objId,shell=True)

            print 'sh %s'%os.path.join(ROOT,'scripts','prepare_ini_new.sh')+' -i %d '%obj_id\
                                + objs[obj_id].WriteShellCmd()

            if os.path.exists('./../sound.wav'):
                print 'WAV FOUND!'
            if os.path.exists('./../sound.raw'):
                print 'RAW FOUND!'

            if is_overwrite:
                print "OVERWRITE!!!"
                # subprocess.call('rm *.wav',shell = True)
                # subprocess.call('rm *.raw',shell = True)
                if os.path.exists('./../sound.wav'):
                    subprocess.call('rm ./../sound.wav',shell = True)

            if not os.path.exists('./../sound.wav') or not os.path.exists('./../sound.raw'):
                print 'wav not generated yet, working on it !'
                # subprocess.call('rm *.wav',shell = True)
                # subprocess.call('rm *.raw',shell = True)
                subprocess.call('bash %s'%os.path.join(ROOT,'scripts','prepare_ini_new.sh')+' -i %d '%obj_id\
                                + objs[obj_id].WriteShellCmd(),shell=True)
                if not os.path.exists('continuous_audio1.wav'):
                    subprocess.call('echo %d sound failed! > %d.txt'%(objs[obj_id].objId,objs[obj_id].objId),shell=True)
                subprocess.call('mv *1.wav ./../sound.wav',shell=True)
                subprocess.call('mv *1.raw ./../sound.raw',shell=True)

            subprocess.call('unlink %d.vmap'%objs[obj_id].objId,shell=True)
            subprocess.call('mv moments moments-%d'%(objs[obj_id].objId),shell=True)
            subprocess.call('unlink %d.geo.txt'%objs[obj_id].objId,shell=True)
            subprocess.call('unlink %d.ev'%objs[obj_id].objId,shell=True)
            #subprocess.call('unlink %d.obj'%obj.objId,shell=True)
            #subprocess.call('unlink %d.orig.obj'%obj.objId,shell=True)
        #copy all wav file to parent folder
        #subprocess.call('mv *.wav ./../',shell=True)
        
    if skip_rendering!=True and (not os.path.exists('./../sli.mp4') or is_overwrite):
        print('Start rendering')
        subprocess.call('sh /data/vision/billf/object-properties/sound/sound/primitives/scripts/single_shape/unset.sh',shell=True)
        print('Start rendering')
        #Create cfg for rendering
        blendercfg = ConfigParser.ConfigParser()
        blendercfg.add_section('Objects')
        blendercfg.set('Objects','obj_num','%d'%objnum)
        for obj_id in range(objnum):
            blendercfg.set('Objects','obj_%d'%obj_id,'%s'%objs[obj_id].objPath)
            blendercfg.set('Objects','obj_%d_rot'%obj_id,'[%f,%f,%f,%f]'%tuple(objs[obj_id].rotation))

        blendercfg.add_section('OutPath')
        blendercfg.set('OutPath','outpath','%s'%renderPath)
        blendercfg.add_section('Camera')
        blendercfg.set('Camera','path','%s'%cam.cfgPath)
        blendercfg.add_section('Lighting')
        blendercfg.set('Lighting','path','%s'%lighting.cfgPath)
        blendercfg.add_section('MotionFile')
        blendercfg.set('MotionFile','path','%s'%os.path.join(simFilePath,'motion.dat'))

        with open('blender_render.cfg', 'w+') as configfile:
            blendercfg.write(configfile)

        #unset python path, call blender and source
        blank = os.path.join(ROOT,'scripts','blank.blend')
        blenderScript = os.path.join(ROOT,'scripts','blender_render_scene.py')
        subprocess.call('unset PYTHONPATH',shell = True)
        subprocess.call('%s %s --background --python %s %s %d'%(BLENDER,blank,blenderScript,\
                         os.path.join(simFilePath,'blender_render.cfg'),skip_factor),shell=True)
        # subprocess.call('source ~/.bash_profile',shell = True)
if skip_video!=True:
    actobj_num = []
    for obj in objs:
        if obj.mass!=0:
            actobj_num.append(obj)
    ffmpeg_video_cmd =\
    'ffmpeg -r 30 -pattern_type glob -i \'./render/*.png\' -pix_fmt yuv420p -crf 0 -vcodec libx264 sli.mp4 -y'
    ffmpeg_audio_cmd = 'ffmpeg '
    if len(actobj_num)==1:
        ffmpeg_audio_cmd = 'cp sound.wav merged.wav'
    else:
        for k in range(0,len(actobj_num)):
            ffmpeg_audio_cmd+='-i sound.wav '
        ffmpeg_audio_cmd+= '-filter_complex amix=inputs=%d:duration=longest merged.wav -y'%(len(actobj_num))
    ffmpeg_movie_cmd = 'ffmpeg -i sli.mp4 -i merged.wav -crf 0 -vcodec libx264 result.mp4 -y'
    with cd(resultPath):
        #subprocess.call('rm -rf sli.mp4 merged.wav result.mp4',shell=True)
        print 'calling %s\n'%ffmpeg_video_cmd 
        subprocess.call(ffmpeg_video_cmd,shell=True)
        print 'calling %s\n'%ffmpeg_audio_cmd 
        subprocess.call(ffmpeg_audio_cmd,shell=True)
        print 'calling %s\n'%ffmpeg_movie_cmd
        subprocess.call(ffmpeg_movie_cmd,shell=True)
        
        
