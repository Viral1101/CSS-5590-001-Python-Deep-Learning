running = True
balance = 0

while running:
    transaction = input("Please enter your transaction:\n")
    current = transaction.split(' ')

    if current[0] == "Withdraw":
        balance -= int(current[1])
    elif current[0] == "Deposit":
        balance += int(current[1])

    print(current[1])

    answer = input("Would you like to continue making transactions?\n")
    if answer[0].lower() == 'n':
        running = False

print("Total Balance: $", balance)
