'''import matplotlib.pyplot as plt
import numpy as np

x = np.array([2020, 2021, 2022, 2023, 2024, 2025])
y = np.array([7, 8, 9, 10, 14, 20])

y_fahrenheit = y * 9/5 + 32
y_kelvin = y + 273.15

plt.subplot(1, 3, 1)
plt.plot(x, y_fahrenheit, 'bo-', label='fahrenheit')
plt.xlabel('Years')
plt.ylabel('Temperature in fahrenheit')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(x, y, 'ro-', label='celsius')
plt.title('Temperature in Helsinki in last six years')
plt.xlabel('Years')
plt.ylabel('Temperature in celsius')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(x, y_kelvin, 'bo-', label='kelvin')
plt.xlabel('Years')
plt.ylabel('Temperature in kelvin')
plt.legend()

plt.tight_layout()
plt.savefig('Temperature.png')
plt.show()
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.array([2020,2021,2022,2023,2024,2025])
y = np.array([7,8,9,10,14,20])
plt.bar(x,y,color="blue")
plt.show()