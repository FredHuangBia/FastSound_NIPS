import subprocess

img_dir = '/data/vision/billf/object-properties/sound/sound/primitives/exp/plots/primV3b_nmh_stft_random_7/'
target_path = '/data/vision/billf/object-properties/sound/sound/primitives/www/primV3b_cnnF_soundnet8_pretrainnone_mse1_LR0.001/151/targets.txt'
label_path = '/data/vision/billf/object-properties/sound/sound/primitives/data/primV3b.txt'
shape_dir = '/data/vision/billf/object-properties/sound/sound/primitives/data/hmstudy/shape/img/'
html_path = '/data/vision/billf/object-properties/sound/sound/primitives/www/loss_shape/loss_shape.html'
mv2 = '/data/vision/billf/object-properties/sound/sound/primitives/www/loss_shape/img/'

tf = open(target_path,'r')
lf = open(label_path,'r')
targets = [line.split()[0] for line in tf]
shapes = {line.split()[0]:int(line.split()[1])+1 for line in lf}
tf.close()
lf.close()

for i in range(14):
	cmd = 'cp '+shape_dir+str(i+1)+'.png '+mv2
	# subprocess.call(cmd,shell=True)

html = open(html_path,'w')
html.write('<html>\n')
html.write('<table border="1">\n')
for target in targets:
	html.write('<tr>\n')
	html.write('<td>\n')
	cmd = 'cp '+img_dir+target+'.dist.png '+mv2
	# subprocess.call(cmd,shell=True)
	html.write('''<img src="img/'''+target+'.dist.png'+'''" style="width:300px">\n''')
	html.write('</td>\n')
	html.write('<td>\n')
	cmd = 'cp '+img_dir+target+'.ll.png '+mv2
	# subprocess.call(cmd,shell=True)
	html.write('''<img src="img/'''+target+'.ll.png'+'''" style="width:300px">\n''')
	html.write('</td>\n')
	html.write('<td>\n')
	html.write('''<img src="img/'''+str(shapes[target])+'.png'+'''" style="width:300px">\n''')
	html.write('</td>\n')
	html.write('</tr>\n')


plot_dir = '/data/vision/billf/object-properties/sound/sound/primitives/code/inference/as/plots/'
target_path = open('/data/vision/billf/object-properties/sound/sound/primitives/exp/test_label.txt','r')
targets = [line.split()[0] for line in target_path]
for target in targets:
	cmd = 'cp '+plot_dir+'astest_stft_height_'+target+'.png '+mv2
	subprocess.call(cmd,shell=True)
	html.write('<tr>\n')
	html.write('<td>\n')
	html.write('''<img src="img/astest_stft_height_'''+target+'.png'+'''" style="width:300px">\n''')
	html.write('</td>\n')
	html.write('<td>\n')
	html.write('''<img src="img/'''+str(shapes[target])+'.png'+'''" style="width:300px">\n''')
	html.write('</td>\n')


html.write('</table>\n')
html.write('</html>\n')

