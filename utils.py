import matplotlib.pyplot as plt

def get_plots(train, val, metric, filename):
	plt.figure()
	plt.plot(range(1, len(train) + 1), train, label='Train')
	plt.plot(range(1, len(val) + 1), val, label='Test')
	plt.title(f'Model {metric}')
	plt.ylabel(metric)
	plt.xlabel('epoch')
	plt.legend()
	plt.savefig(filename)
	print(f"saved figure {filename}_{metric}")