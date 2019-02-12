tuple_list = list()
tuple_list.append(('John',('Physics', 80)))
tuple_list.append(('Daniel',('Science', 90)))
tuple_list.append(('John',('Science', 95)))
tuple_list.append(('Mark',('Maths', 100)))
tuple_list.append(('Daniel',('History', 75)))
tuple_list.append(('Mark',('Social', 95)))

dictionary = {}
for entry in tuple_list:
    if entry[0] not in dictionary.keys():
        dictionary[entry[0]] = [entry[1]]
    else:
        dictionary[entry[0]].append(entry[1])

print(dictionary)
