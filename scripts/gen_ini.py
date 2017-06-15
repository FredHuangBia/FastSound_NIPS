f = open("time.ini","w")
for i in range(200):
	f.write(str(i/float(201)*2)+" ")
