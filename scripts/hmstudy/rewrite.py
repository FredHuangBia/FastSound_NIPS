ROOT = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/'

bad = [1,3,8,9,14,16,17,18,20,25,29,30,33,35,41,46,47,52,53,54,55,58,62,65,66,68,70,72,76,78,81,85,86,88,92,93,98]

txt_all = open(ROOT+'001-100.txt','r').readlines()
dat_all = open(ROOT+'001-100.dat','r').readlines()

txt_new = open(ROOT+'001-100_new.txt','w')
dat_new = open(ROOT+'001-100_new.dat','w')

for i in range(len(txt_all)):
	txt_line = int(txt_all[i].split()[0])
	if txt_line in bad:
		tmp_txt = open(ROOT+'%03d-%03d.txt'%(txt_line,txt_line),'r').readlines()[0].strip()
		txt_new.write('%s\n'%tmp_txt)
		tmp_dat = open(ROOT+'%03d-%03d.dat'%(txt_line,txt_line),'r').readlines()[0].strip()
		dat_new.write('%s\n'%tmp_dat)
	else:
		tmp_txt = txt_all[i].strip()
		txt_new.write('%s\n'%tmp_txt)
		tmp_dat = dat_all[i].strip()
		dat_new.write('%s\n'%tmp_dat)
