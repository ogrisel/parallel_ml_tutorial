_ = plt.hist(scores, range=(0, 1), bins=30, alpha=0.2)

from scipy.stats.kde import gaussian_kde
x = np.linspace(0, 1, 1000)
smoothed = gaussian_kde(scores).evaluate(x)
plt.plot(x, smoothed, label="Smoothed distribution")

top = np.max(smoothed)
plt.vlines([np.mean(scores)], 0, top, color='r', label="Mean test score")
plt.vlines([np.median(scores)], 0, top, color='b', linestyles='dashed',
           label="Median test score")
plt.legend(loc='best')
_ = plt.title("Cross Validated test scores distribution")
