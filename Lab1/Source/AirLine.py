import numpy as np


# The Person class defines the base class that Employees and Passengers inherit attributes from
class Person(object):

    def __init__(self, first, last, luggage):
        self.first_name = first
        self.last_name = last
        self.luggage_list = luggage

    def set_first(self, first):
        self.first_name = first

    def set_last(self, last):
        self.last_name = last

    def set_luggage(self, luggage):
        self.luggage_list = luggage

    def get_first(self):
        return self.first_name

    def get_last(self):
        return self.last_name

    def get_luggage(self):
        return self.luggage_list

    def get_name(self):
        return "%s, %s" % (self.last_name, self.first_name)  # Concatenate the first and last name


# Emplopyee class adds the position attribute to a Person
class Employee(Person):

    def __init__(self, first, last, luggage, position):
        super().__init__(first, last, luggage)  # Uses super() as a shortcut for referencing the Person class
        self.position = position

    def set_position(self, position):
        self.position = position

    def get_position(self):
        return self.position

    def __str__(self):
        return "%s is serving as %s for this flight." % (self.get_name(), self.position)


# Passenger class adds the seat and seat type attributes to a Person
class Passenger(Person):

    def __init__(self, first, last, luggage, seat_class, seat):
        super().__init__(first, last, luggage)  # Uses super() as a shortcut for referencing the Person class
        self.seat_class = seat_class
        self.seat = seat

    def set_seat_class(self, seat_class):
        self.seat_class = seat_class

    def set_seat(self, seat):
        self.seat = seat

    def get_seat_class(self):
        return self.seat_class

    def get_seat(self):
        return self.seat

    def __str__(self):
        return "%s is flying %s in seat %s" % (self.get_name(), self.seat_class, self.seat)


# The manifest class stores a collection of both the passengers and crew members.
# It allows the creation of new Crew members and Passengers as well as their removal.
# It also keeps a tally of the total number of people in the manifest.
class Manifest(object):
    ON_BOARD = 0

    def __init__(self):
        self.__crew: [Employee] = []            # Type the contents of the list
        self.__passengers: [Passenger] = []     # Type the contents of the list

    def add_crew(self, first, last, luggage, position):
        self.__crew.append(Employee(first, last, luggage, position))
        Manifest.ON_BOARD = Manifest.ON_BOARD + 1       # Increment the manifest count

    def add_passenger(self, first, last, luggage, seat_class, seat):
        self.__passengers.append(Passenger(first, last, luggage, seat_class, seat))
        Manifest.ON_BOARD = Manifest.ON_BOARD + 1       # Increment the manifest count

    def remove_crew(self, first, last):
        for i in self.__crew:
            if i.get_name() == "%s, %s" % (last, first):
                self.__crew.remove(i)
                Manifest.ON_BOARD = Manifest.ON_BOARD - 1   # Decrement the manifest count

    def remove_passenger(self, first, last):
        for i in self.__passengers:
            if i.get_name() == "%s, %s" % (last, first):
                self.__passengers.remove(i)
                Manifest.ON_BOARD = Manifest.ON_BOARD - 1   # Decrement the manifest count

    def __str__(self):
        crew_string = ""
        passenger_string = ""

        for x in self.__crew:
            crew_string = crew_string + "%s\n" % x.__str__()

        for y in self.__passengers:
            passenger_string = passenger_string + "%s\n" % y.__str__()

        return "There are %d souls on board.\n\nServing as crew:\n%s\n\nPassenger list:\n%s" % \
               (Manifest.ON_BOARD, crew_string, passenger_string)


# Route defines the parameters of the flight
class Route(object):

    def __init__(self, start, destination, depart, arrive):
        self.__start = start
        self.__destination = destination
        self.__departure_stamp = depart
        self.__arrival_stamp = arrive

    def set_start(self, start):
        self.__start = start

    def set_destination(self, dest):
        self.__destination = dest

    def set_departure(self, depart):
        self.__departure_stamp = depart

    def set_arrival(self, arrival):
        self.__arrival_stamp = arrival

    def get_start(self):
        return self.__start

    def get_destination(self):
        return self.__destination

    def get_departure(self):
        return self.__departure_stamp

    def get_arrival(self):
        return self.__arrival_stamp

    def __str__(self):
        return "departs from %s at %s and is due to arrive in %s at %s." % \
               (self.__start, self.__departure_stamp, self.__destination, self.__arrival_stamp)


# Plane defines the attributes of the plane being used for a particular Flight
class Plane(object):

    def __init__(self, model, capacity):
        self.__model = model
        self.__capacity = capacity

    def set_model(self, model):
        self.__model = model

    def set_capacity(self, capacity):
        self.__capacity = capacity

    def get_model(self):
        return self.__model

    def get_capacity(self):
        return self.__capacity

    def __str__(self):
        return "Plane model: %s carrying capacity: %d persons" % (self.__model, self.__capacity)


# The Flight class inherits properties from Route, Plane, and Manifest
# The manifest is empty by default and will be populated later.
# The flight will have a predetermined schedule and plane model for the flight in order to build a manifest
# Therefore, route and plane will be intialized at obejct instantiation.
class Flight(Route, Plane, Manifest):

    def __init__(self, start, destination, depart, arrive, model, capacity):
        Manifest.__init__(self)
        Route.__init__(self, start, destination, depart, arrive)
        Plane.__init__(self, model, capacity)
        self.__id = np.random.randint(0, 4000 + 1, 1)

    def __str__(self):
        return "Flight No: %d\n%s\n%s\n%s" % (self.__id, Route.__str__(self), Plane.__str__(self), Manifest.__str__(self))


# Instantiate an object with Route and Plane parameters
a = Flight("MCI", "LAX", "15:34 14Feb2019", "17:14 14Feb2019", "Airbus A319", 160)

# Add People to the manifest; both crew and passengers
a.add_crew("Jim", "Jones", 1423, "Pilot")
a.add_crew("Larry", "Llewynn", 1687, "Pilot")
a.add_crew("Cathy", "Catzman", 3416, "Flight Attendant")
a.add_crew("Tom", "Thomas", 1463, "Flight Attendant")
a.add_crew("Frank", "Franklin", 8142, "Flight Attendant")
a.add_passenger("Lucy", "Liu", 2875, "First Class", "1A")
a.add_passenger("Bernie", "Sanders", 1684, "Economy", "17C")

# Print out information about the flight, including the manifest
print(a.__str__())
