import os
import subprocess

obj_path = "/data/vision/billf/object-properties/sound/sound/primitives/shapes/obj/"
ply_path = "/data/vision/billf/object-properties/sound/sound/primitives/shapes/ply/"
save_path = "/data/vision/billf/object-properties/sound/sound/primitives/shapes/all/"



# obj_list = os.listdir(obj_path)

# map_file = open(obj_path+"map.txt","w")

# current = 0

# for obj in obj_list:
# 	map_file.write(str(current)+" "+obj+"\n")
# 	subprocess.call(["cp", obj_path+obj, save_path+str(current)+".obj"])
# 	current+=1

# map_file.close()



map_file = open(save_path+"map.txt","r")

for line in map_file:
	line = line.split()
	obj_name = line[1]
	index = line[0]
	subprocess.call(["cp", ply_path+obj_name[0:-4]+'.ply', save_path+index+".ply"])

map_file.close()