''' 1- Draw the lines y=2x+1, y=2x+2, y=2x+3
in the same figure.
Use different drawing colors and line types for your graphs to make them stand out in black and white. Set the image title and labels for the horizontal and vertical axes.
'''
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(0,10,100)
y1 = 2*x+1
y2 = 2*x+2
y3 = 2*x+3
plt.plot(x,y1,'r-',label='y=2x+1')
plt.plot(x,y2,'g--',label='y=2x+2')
plt.plot(x,y3,'b-.',label='y=2x+3')
plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.title('Graphs of y=2x+1, y=2x+2, y=2x+3')
plt.legend()
plt.show()


'''Create a numpy vector x that contains the values 1,2,3,4,5,6,7,8,9.
Create another vector y with the values −0.57,−2.57,−4.80,−7.36,−8.78,−10.52,−12.85,−14.69,−16.78.
Draw the scattering pattern of the points (x,y). Use the + symbol to represent points.'''

x = np.array([1,2,3,4,5,6,7,8,9])
y = np.array([-0.57,-2.57,-4.80,-7.36,-8.78,-10.52,-12.85,-14.69,-16.78])
plt.scatter(x,y,marker='+')
plt.show()

'''Read from CSV-file weight-height.csv to numpy-table information about the lengths and weights (in inches and pounds) of a group of students. Collect the lengths for the variable length and the weights for the variable weight by cutting the table.
Convert lengths from inches to centimeters and weights from pounds to kilograms.
Calculate the means of the lengths and weights.
Finally draw a histogram of the lengths.'''
# Load data, handling mixed types and skipping headers
filename = "weight-height.csv"
data = np.genfromtxt(filename, delimiter=',', skip_header=1, usecols=(1, 2))  # Assuming length and weight are in cols 1 & 2

# Extract length (height) and weight columns
length = data[:, 0]  # Height in inches
weight = data[:, 1]  # Weight in pounds

# Convert lengths to centimeters and weights to kilograms
length_cm = length * 2.54  # 1 inch = 2.54 cm
weight_kg = weight * 0.453592  # 1 pound = 0.453592 kg

# Calculate means
mean_length = np.mean(length_cm)
mean_weight = np.mean(weight_kg)

print(f"Mean length (cm): {mean_length:.2f}")
print(f"Mean weight (kg): {mean_weight:.2f}")

# Plot histogram of lengths
plt.hist(length_cm, bins=20, color='blue', edgecolor='black', alpha=0.7)
plt.xlabel("Length (cm)")
plt.ylabel("Frequency")
plt.title("Histogram of Student Lengths")
plt.grid(True)
plt.show()

'''Calculate the inverse matrix of matrix A and check with the matrix product that both AA−1 and A−1 A produce a unit matrix with ones in diagonals and zeros elsewhere (the values will not be exactly 1 and 0 due to floating point error).
A=([[1,2,3],
         [0,1,4],
         [5,6,0]])'''


A = np.array([[1,2,3],[0,1,4],[5,6,0]])
A_inv = np.linalg.inv(A)
print(A_inv)
print(np.dot(A,A_inv))
print(np.dot(A_inv,A))




