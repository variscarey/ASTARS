import matplotlib.pyplot as plt
import numpy as np

params = {'legend.fontsize': 28,'legend.handlelength': 3}
plt.rcParams["figure.figsize"] = (60,40)
plt.rcParams['figure.dpi'] = 80
plt.rcParams['savefig.dpi'] = 100
plt.rcParams['font.size'] = 30
plt.rcParams['figure.titlesize'] = 'xx-large'
plt.rcParams.update(params)

stars_full, sf_ls = 'red', '--'
active_stars_learned, lr_ls = 'black', '-.'
active_stars_ref, rf_ls = 'blue', ':'

### import data

data = np.load('nltest.npz')
act = data['sub']
inact = data['inact']
xdata = data['x']
fdata = data['fx']

stars_data = np.load('nltest_stars.npz')

x_sdata = stars_data['x']
f_sdata = stars_data['fx']

plt.semilogy(fdata,label = 'ASTARS, Global Linear Surrogate', lw = 5, color = active_stars_learned, ls = lr_ls)
plt.semilogy(f_sdata, label = 'STARS', color = stars_full, lw = 5, ls = sf_ls)
plt.legend()
plt.xlabel('$k$, iteration count')
plt.ylabel('$|$ data misfit $|$')
plt.savefig('dci_nl_xcube.png')
plt.show()
