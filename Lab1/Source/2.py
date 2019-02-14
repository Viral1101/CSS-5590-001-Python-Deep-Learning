# Initialize list
tuple_list = list()
tuple_list.append(('John',('Physics', 80)))
tuple_list.append(('Daniel',('Science', 90)))
tuple_list.append(('John',('Science', 95)))
tuple_list.append(('Mark',('Maths', 100)))
tuple_list.append(('Daniel',('History', 75)))
tuple_list.append(('Mark',('Social', 95)))

# Create empty Dictionary
dictionary = {}

# Iterate through each item in list
for entry in tuple_list:
    # If name isn't already a key, create new key value pair
    if entry[0] not in dictionary.keys():
        dictionary[entry[0]] = [entry[1]]
    # If name is already in dictionary, add the new score to the value
    else:
        dictionary[entry[0]].append(entry[1])
        dictionary[entry[0]].sort()

print(dictionary)
