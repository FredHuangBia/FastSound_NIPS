import matplotlib.pyplot as plt

def log_parser(logfile):
	ll = []
	for line in open(logfile,'r'):
		if len(line.split()) > 1:
			if line.split()[0] == 'Initial':
				ll.append(float(line.split()[3]))
			if line.split()[0] == '#####':
				assert(len(ll) == int(line.split()[2])), "Something Wrong!"
				ll.append(float(line.split()[4]))
	return ll

def main():
	nn_init = log_parser('abs.log')
	nn_init2 = log_parser('abs2.log')
	nn_init3 = log_parser('abs3.log')
	rdm_init = log_parser('abs_random.log')
	rdm_init2 = log_parser('abs_random2.log')
	rdm_init3 = log_parser('abs_random3.log')

	plt.plot(range(len(nn_init)),nn_init, 'b', range(len(rdm_init)),rdm_init, 'y')
	plt.plot(range(len(nn_init2)),nn_init2, 'b', range(len(rdm_init2)),rdm_init2, 'y')
	plt.plot(range(len(nn_init3)),nn_init3, 'b', range(len(rdm_init2)),rdm_init3, 'y')
	# plt.show()
	plt.savefig("exp.png",dpi=300,bbox_inches='tight')

if __name__ == '__main__':
	main()