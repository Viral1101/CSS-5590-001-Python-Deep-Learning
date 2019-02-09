import numpy as np

# Build a vector of length 15 with integers from 0 to 20 inclusive
a = np.random.randint(0, 20 + 1, 15)
print(a)  # Show the vector

a.sort()  # Order the vector

# Holder variables
nmax = 0
count = 0
mode = 0

# Iterate through the sorted list and count like entries
# When the count exceeds the max and the number in the list increases
# Save the current max as the count
y = a[0]
for x in a:
    if x > y:
        if count > nmax:
            nmax = count
            mode = y
        count = 1
        y = x
    else:
        count = count + 1

# Check for the final number since it isn't covered by the for loop
if count > nmax:
    nmax = count
    mode = y

# Results
print("The mode is %d. It appears %d times." % (mode, nmax))

