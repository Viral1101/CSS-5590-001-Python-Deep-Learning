string = input("Please enter Python students separated by a space.\n")
python = string.split(" ")
string = input("Please enter Web Application students separated by a space.\n")
web_application = string.split(" ")

set_python = set(python)
set_web_application = set(web_application)
set_both = set_python & set_web_application
set_not_common = set_python ^ set_web_application

both = list(set_both)
not_common = list(set_not_common)

print("Python Students: ", python)
print("Web Application Students: ", web_application)
print("Both Course Students: ", both)
print("Students in One Course: ", not_common)
