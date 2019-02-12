# Initialize
running = True
balance = 0

# Loop to get transactions
while running:
    transaction = input("Please enter your transaction:\n")
    current = transaction.split(' ')

    # If withdraw, remove amount from balance
    if current[0] == "Withdraw":
        balance -= int(current[1])
    # If deposit, add amount to balance
    elif current[0] == "Deposit":
        balance += int(current[1])

    answer = input("Would you like to continue making transactions?\n")
    # If user answers no, quit looping
    if answer[0].lower() == 'n':
        running = False

print("Total Balance: $", balance)
