import control as ct
import matplotlib.pyplot as plt

# Example: G(s) = 5 / (s^2 + 2*s + 5)
num = [5]
den = [1, 2, 5]
G = ct.TransferFunction(num, den)

t, y = ct.step_response(G)
plt.figure()
plt.plot(t, y)
plt.title('Step Response')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()