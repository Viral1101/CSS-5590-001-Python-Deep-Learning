# Find and display the numbers with 3 odd digits in the range 100 to 500

# Set an empty list to collect the numbers
odds = []

# Iterate through the numbers 100-500
for x in range(100, 500 + 1):
    odd = True      # Set the default condition to be true
    nums = str(x)   # Convert the numbers to strings
    for y in nums:      # Iterate through each 'digit'
        num = int(y)        # Convert the digit back to an integer
        if num % 2 == 0:    # Check if the number is even
            odd = False     # Set the condition to false if the number is even
    if odd:
        odds.append(x)  # Append the number to the list if all three digits are odd

print(odds)

# Sort a predefined list of strings
list=["1", "4", "0", "6", "9"]
list.sort()
print(list)

# Count the number of words and letters per line in a read file

infile = open("icp.txt", 'r') # Load a text file

lines = infile.readlines()  # Create a list of the lines
for x in lines:             # Iterate over the lines
    chars = 0                   # Initialize a the count to 0
    words = len(x.split(" "))   # Count the number of words in the line
    for y in x.split(" "):      # Iterate over the words
        for i in y:                 # Iterate over the characters in the word
            if i.isalpha():             # Increment the count if the character is alphabetical
                chars = chars + 1
    print("%s |Words: %d |Letters: %d |" % (x.rstrip('\n'), words, chars)) # String format to print the output