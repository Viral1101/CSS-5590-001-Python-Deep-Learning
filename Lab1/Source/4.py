# Get the input string
string = input("Please enter a string:\n")

# Variable to hold the longest substring
best_chars = [string[0]]
# Variable to hold the current substring
current_chars = [string[0]]
# Initially the first character is already selected for the best substring
current_length = 1
best_length = 1

# Iterate through the given string, except the first character
for i in range(1, len(string)):
    # If next character is not in current, add to current substring and increase length by 1
    if string[i] not in current_chars:
        current_length += 1
        current_chars.append(string[i])
    # If next character is in current, update the best substring if necessary
    else:
        if current_length > best_length:
            best_chars = current_chars
            best_length = current_length
        # While the current character is in the current substring, remove letters from the beginning
        while string[i] in current_chars:
            current_chars = current_chars[1:]
            current_length -= 1
        # Finally add the current character to the substring and increase length
        current_chars.append(string[i])
        current_length += 1

# Convert list of chars to a string
best_string = ""
for char in best_chars:
    best_string += char

# Print results
print("Longest Substring with No Repeating Characters:", best_string)
print("Length: ", best_length)
