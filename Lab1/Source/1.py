# Initialize
balance = 0

# Loop to get transactions
lines = []
print("Please paste your transaction log:\n")
while True:
    line = input()
    if line == '':
        break
    else:
        lines.append(line)

for x in lines:

    transaction = x
    current = transaction.split(' ')

    # If withdraw, remove amount from balance
    if current[0] == "Withdraw":
        balance -= float(current[1])
    # If deposit, add amount to balance
    elif current[0] == "Deposit":
        balance += float(current[1])

    # answer = input("Would you like to continue making transactions?\n")
    # # If user answers no, quit looping
    # if answer[0].lower() == 'n':
    #     running = False

print("Total Balance: $%.2f" % balance)
