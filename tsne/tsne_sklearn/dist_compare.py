# This is not my creation
# It is simply the mimic of the following blog post from oreilly
# https://www.oreilly.com/learning/an-illustrated-introduction-to-the-t-sne-algorithm

import numpy as np
import matplotlib.pyplot as plt

z = np.linspace(0., 5., 1000)
gauss = np.exp(-z**2)
cauchy = 1/(1+z**2)
plt.plot(z, gauss, label='Gaussian')
plt.plot(z, cauchy, label='Cauchy')
plt.legend()
plt.savefig('images/distribution-generated.png', dpi=100)
