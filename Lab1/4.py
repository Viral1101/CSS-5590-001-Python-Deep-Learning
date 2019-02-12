string = input("Please enter a string:\n")

best_chars = [string[0]]
current_chars = [string[0]]
first = 0
last = 0
current_length = 1
best_length = 1

for i in range(1, len(string)):
    if string[i] not in current_chars:
        current_length += 1
        current_chars.append(string[i])
        last = i
    else:
        if current_length > best_length:
            best_chars = current_chars
            best_length = current_length
        while string[i] in current_chars:
            current_chars = current_chars[1:]
            current_length -= 1
            first += 1
        current_chars.append(string[i])
        current_length += 1
        last = i

best_string = ""
for char in best_chars:
    best_string += char

print("Longest Substring with No Repeating Characters:", best_string)
print("Length: ", best_length)
