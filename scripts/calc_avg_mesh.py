import os

path = "/data/vision/billf/object-properties/sound/sound/primitives/data/v2.3/"



ptmean=0
ptmax=0
ptmin=1000000

tetmean=0
tetmax=0
tetmin=1000000

fcmean=0
fcmax=0
fcmin=1000000

ffmean=0
ffmax=0
ffmin=1000000

good=0
total=0

for i in range(0,14,1):
	# print(i)
	file = path+str(i)+"/tetgen.log"
	f = open(file)
	for line in f:
		line = line.split()
		if len(line)<1:
			continue
		if line[0]=="Mesh":
			if line[1]=="points:":
				total +=1
				ptmean = (ptmean*(total-1)+int(line[2]))/float(total)
				if int(line[2])>ptmax:
					ptmax = int(line[2])
				if int(line[2])<ptmin:
					ptmin = int(line[2])
			elif line[1]=="tetrahedra:":
				tetmean = (tetmean*(total-1)+int(line[2]))/float(total)
				if int(line[2])>tetmax:
					tetmax = int(line[2])
				if int(line[2])<tetmin:
					tetmin = int(line[2])
					print(i)
			elif len(line)==3 and line[1]=="faces:":
				fcmean = (fcmean*(total-1)+int(line[2]))/float(total)
				if int(line[2])>fcmax:
					fcmax = int(line[2])
				if int(line[2])<fcmin:
					fcmin = int(line[2])
			elif len(line)==5 and line[1]=="faces":
				ffmean = (ffmean*(total-1)+int(line[4]))/float(total)
				if int(line[4])>ffmax:
					ffmax = int(line[4])
				if int(line[4])<ffmin:
					ffmin = int(line[4])
				if int(line[4])<=1500:
					good+=1
			else:
				pass
		else:
			pass

print("point mean: ",ptmean,"\npoint max: ",ptmax,"\npoint min: ",ptmin, "\n")

print("tet mean: ",tetmean,"\ntet max: ",tetmax,"\ntet min:",tetmin, "\n")

print("faces mean: ",fcmean,"\nfaces max: ",fcmax,"\nfaces min:",fcmin,"\n")

print("faces on facets mean: ",ffmean,"\nfaces on facets max: ",ffmax,"\nfaces on facets min:",ffmin,"\n")

print("selected: ",good,"\n")

print("total: ",total,"\n")
