import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("aaplCombined.csv")
# Calculate the 5-day moving average
data['5-day MA'] = data['Close'].rolling(window=5).mean()
data = data.dropna()
# Split the data into training and test sets
train_data = data[:1513]
test_data = data[1513:]


# Initialize the cash and stock holdings
cash_held = 10000
stocks_held = 25
initial_portfolio_value = 10000
remaining_cash = 10000 - test_data['Open'][0] * stocks_held

# Initialize portfolio value, cash, and stocks held
portfolio_value = initial_portfolio_value
cash_held = remaining_cash
stocks_held = 25

portfolio_values = []

# Iterate through each day in the test dataset
for i in range(len(test_data)):
    # Define the state for the current day
    state = np.array([test_data['Open'][i], test_data['5-day MA'][i], portfolio_value, cash_held, stocks_held])

    # Take an action using the trained DQN model
    Q_values = model.predict(np.array([state]))
    action = np.argmax(Q_values)

    # Calculate the reward for the action taken
    price_change_percentage = ((test_data['Open'][i] - test_data['5-day MA'][i]) / test_data['5-day MA'][i]) * 100
    reward = price_change_percentage

    # Update the cash and stock holdings based on the action taken
    if action == 0:
        stocks_bought = cash_held // test_data['Open'][i]
        cash_held -= stocks_bought * test_data['Open'][i]
        stocks_held += stocks_bought
        print("buy")
    elif action == 1:
        cash_gained = stocks_held * test_data['Open'][i]
        cash_held += cash_gained
        stocks_held = 0
        print("sell")
    elif action == 2:
        print("hold")

    # Update the portfolio value with the current cash and stock holdings
    portfolio_value = cash_held + stocks_held * test_data['Open'][i]
    portfolio_values.append(portfolio_value)
#print portfolio value
print(portfolio_value)
# Plot the portfolio value over time
plt.plot(range(len(test_data)), portfolio_values)
plt.xlabel('Day')
plt.ylabel('Portfolio Value')
plt.show()
