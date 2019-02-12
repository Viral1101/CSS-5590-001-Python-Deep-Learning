# Get starting input for each class roster
string = input("Please enter Python students separated by a space.\n")
python = string.split(" ")
string = input("Please enter Web Application students separated by a space.\n")
web_application = string.split(" ")

# Create sets of each for easy set arithmetic
set_python = set(python)
set_web_application = set(web_application)

# Find the intersection of each class
set_both = set_python & set_web_application
# Find the symmetric difference for the classes
set_not_common = set_python ^ set_web_application

# Convert sets back to lists
both = list(set_both)
not_common = list(set_not_common)

# Print all
print("Python Students: ", python)
print("Web Application Students: ", web_application)
print("Both Course Students: ", both)
print("Students in One Course: ", not_common)
