# Exp 7: Multipath fading - sum of many reflected paths (Rayleigh) as receiver moves
import numpy as np, matplotlib.pyplot as plt
np.random.seed(1)
f, c = 900e6, 3e8; lam = c / f                 # 900 MHz carrier
d = np.linspace(0, 5, 1000)                    # receiver moves 5 m
P = 20                                         # number of paths
amp = np.random.rand(P); ph = 2*np.pi*np.random.rand(P); ang = 2*np.pi*np.random.rand(P)
# each path has a different phase shift depending on arrival angle -> add them up
h = sum(amp[k]*np.exp(1j*(ph[k] + 2*np.pi*d/lam*np.cos(ang[k]))) for k in range(P))
rssi = 20*np.log10(np.abs(h)/np.abs(h).mean())  # dB relative to mean

plt.figure(figsize=(9, 4))
plt.plot(d, rssi); plt.axhline(-10, color="r", ls="--", label="deep fade (-10 dB)")
plt.xlabel("Distance moved (m)"); plt.ylabel("Signal level (dB)"); plt.grid(alpha=.3); plt.legend()
plt.title("Multipath Fading: 20 reflected paths, 900 MHz")
plt.tight_layout(); plt.savefig("exp7_multipath.png", dpi=150); plt.show()
dips = np.sum((rssi[1:-1] < rssi[:-2]) & (rssi[1:-1] < rssi[2:]) & (rssi[1:-1] < -10))   # separate deep fades
print("OBSERVATION TABLE")
print("Max signal      : %+.1f dB at %.2f m" % (rssi.max(), d[rssi.argmax()]))
print("Min signal      : %+.1f dB at %.2f m" % (rssi.min(), d[rssi.argmin()]))
print("Variation       : %.1f dB (max - min)" % (rssi.max() - rssi.min()))
print("Time below -10 dB: %.1f %%" % (100*(rssi < -10).mean()))
print("Deep fades (<-10 dB): %d in 5 m   (wavelength = %.2f m)" % (dips, lam))
