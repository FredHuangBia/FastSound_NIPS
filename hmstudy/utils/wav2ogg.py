import subprocess
root = '/data/vision/billf/object-properties/sound/sound/primitives/www/amt_fastsound/pose/data/sound/'

#this is the original root of all the test sond, labels are in the corresponding folders
from_path = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudyV4a_height/'

# ffmpeg -i 1.wav -c:a libfdk_aac -b:a 128k 1.m4a 
# ffmpeg -i 1.wav  -acodec libvorbis 1.ogg 

for i in range(50):
	cmd = 'cp '+from_path+str(i)+'/sound.wav '+root+str(i)+'.wav'
	subprocess.call(cmd,shell=True)
	cmd = 'ffmpeg -i ' + root+str(i) + '.wav -c:a libfdk_aac -b:a 128k ' + root+str(i) + '.m4a'
	subprocess.call(cmd,shell=True)
	cmd = 'ffmpeg -i ' + root+str(i) + '.wav -acodec libvorbis ' + root + str(i) + '.ogg'
	subprocess.call(cmd,shell=True)