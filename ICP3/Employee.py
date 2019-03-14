class Employee(object):
    EMPLOYEE_COUNT = 0  # Keep running tally of number of employees

    # Initialize the object attributes
    def __init__(self, first, last, pay, dept):
        self.first = first
        self.last = last
        self.salary = pay
        self.dept = dept
        Employee.EMPLOYEE_COUNT = Employee.EMPLOYEE_COUNT + 1  # increment the count when initializing an employee


    # Getters and setters

    def setFirstName(self, name):
        self.first = name

    def setLastName(self, name):
        self.last = name

    def setSalary(self, pay):
        self.salary = pay

    def setDept(self, dept):
        self.dept = dept

    def getName(self):
        return "%s, %s" % (self.last, self.first)  # Concatenate the first and last name

    def getPay(self):
        return self.salary

    def getDept(self):
        return self.dept

    #To string function
    def __str__(self):
        return "%s earns $%.2f per hour and works in the the %s department." % \
               (self.getName(), self.getPay(), self.getDept())


class FullTimeEmployee(Employee):

    # Initialize a FullTimeEmployee in terms of the Employee class
    def __init__(self,first, last, pay, dept, medical, retirement):
        Employee.__init__(self, first, last, pay, dept)
        self.medical = medical
        self.retirement = retirement

    # Getters and setters
    def getMedical(self):
        return self.medical

    def getRetirement(self):
        return self.retirement

    def setMedical(self, medical):
        self.medical = medical

    def setRetirement(self, retirement):
        self.retirement = retirement

    # To string function
    def __str__(self):
        return "%s Employee is Fulltime on Medical plan %s and saving %.2f percent for retirement." % \
               (Employee.__str__(self), self.getMedical(), self.getRetirement())


print(Employee.EMPLOYEE_COUNT)  # Count should be 0

# Initialize some employees
a = Employee("Klark", "Kent", 15.46, "Cape")
b = Employee("Lois", "Lane", 17.47, "Print")
print(Employee.EMPLOYEE_COUNT)  # Count should be 2

# Initialize some full time employees
c = FullTimeEmployee("Peter", "Parker", 21.16, "Photo", "Plan A", 5)
d = FullTimeEmployee("Nick", "Fury", 29.74, "Photo", "Plan C", 7)

# Collect all employees into a list
empList = [a, b, c, d]

# Define function to average employee salaries
def average(emplist):

    total = 0
    for i in emplist:
        total = total + i.getPay()

    return total / len(emplist)


# Show all employee info
for x in empList:
    print(x.__str__())

# Utilize some class methods
a.setFirstName("Clark")
b.setLastName("Kent")
c.setMedical("C")
c.setSalary(16.21)
d.setRetirement(8)

# Compute and show average salary
print("Average salary is %.2f" % (average(empList)))


# Show the changed employee info
for x in empList:
    print(x.__str__())

print(Employee.EMPLOYEE_COUNT)  # Count should be 4
