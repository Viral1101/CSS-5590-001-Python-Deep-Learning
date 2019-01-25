print ("The differences between python 2 and 3:")
print ("-print is now a function and requires ()")
print ("-range() is lazy, it is xrange() from python 2")
print ("-integer division gives a floating point result in 3\n")

number = input("Enter a number to reverse:")
reverse = ""
for x in number:
    reverse = x + reverse
print(reverse + "\n")

number1 = int(input("Enter the first number: "))
number2 = int(input("Enter the second number: "))

add = number1 + number2
subtract = number1 - number2
multiply = number1 * number2
divide = number1/number2

print("%.2f + %.2f = %.2f" % (number1, number2, add))
print("%.2f - %.2f = %.2f" % (number1, number2, subtract))
print("%.2f * %.2f = %.2f" % (number1, number2, multiply))
print("%.2f / %.2f = %.2f\n" % (number1, number2, divide))

sentence = input("Enter a sentence: ")
words = sentence.split(" ")
numDigits = 0
numLetters = 0
numWords = len(words)
for x in sentence:
    if x.isalpha():
        numLetters = numLetters + 1
    if x.isdigit():
        numDigits = numDigits + 1
print("Number of words: %d | Number of letters: %d | Number of digits: %d" % (numWords, numLetters, numDigits))