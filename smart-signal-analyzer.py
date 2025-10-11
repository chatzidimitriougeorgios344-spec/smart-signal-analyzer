import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# ----- Δημιουργία σήματος -----
def generate_signal(fs=2000, duration=1.0):
    t = np.arange(0, duration, 1/fs)
    f1, f2 = 50, 400
    signal_clean = 1.2*np.sin(2*np.pi*f1*t) + 0.7*np.sin(2*np.pi*f2*t)
    transient = signal.chirp(t, f0=100, f1=800, t1=0.4, method='linear') * np.exp(-5*(t-0.2)**2)
    noise = 0.6 * np.random.normal(0, 1, len(t))
    x = signal_clean + 0.8*transient + noise
    return t, x

# ----- Φίλτρο Butterworth -----
def butter_bandpass_filter(x, fs, lowcut=20, highcut=120, order=4):
    nyq = 0.5 * fs
    low, high = lowcut/nyq, highcut/nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return signal.filtfilt(b, a, x)

# ----- Υπολογισμός FFT -----
def plot_fft(x, fs, title="FFT Magnitude"):
    N = len(x)
    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    mag = np.abs(X)/N
    plt.figure(figsize=(8,4))
    plt.semilogy(freqs, mag)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    plt.tight_layout()
    plt.show()

# ----- Κύριο πρόγραμμα -----
def main():
    fs = 2000
    t, x = generate_signal(fs)
    x_filt = butter_bandpass_filter(x, fs)

    N = len(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    X = np.abs(np.fft.rfft(x))/N
    Xf = np.abs(np.fft.rfft(x_filt))/N

    plt.figure(figsize=(10,6))
    plt.subplot(2,1,1)
    plt.semilogy(freqs, X)
    plt.title("Raw Signal FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.subplot(2,1,2)
    plt.semilogy(freqs, Xf)
    plt.title("Filtered Signal FFT")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")

    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()

