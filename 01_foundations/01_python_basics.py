"""
Python Basics for Machine Learning
Complete these exercises to get comfortable with Python
"""

# ================================================
# Exercise 1: Variables and Data Types
# ============================================
print("=== Exercise 1: Variables ===")

# TODO: Create variables for:
# - Your name (string)
# - Your age (integer)
# - Your height in meters (float)
# - Whether you like ML (boolean)

name = "Srinidhi"
age = 26
height = 70 # in inches
likes_ml = True

print(f"Name: {name}, Age: {age}, Height: {height}m, Likes ML: {likes_ml}")

# ============================================
# Exercise 2: Lists and Dictionaries
# ============================================
print("\n=== Exercise 2: Data Structures ===")

# TODO: Create a list of 5 numbers
numbers = [1, 2, 3, 4, 5]

# TODO: Create a dictionary with your info
info = {
    "first_name": "Srinidhi",
    "last_name": "SP",
    "age": 26,
    "city": "San Francisco",
    "country": "United States",
    "occupation": "Data Engineer",
    "friends": ["Aparna", "Anand", "Arun", "Arunima", "Aravind"]
}

print(f"Numbers: {numbers}")
print(f"City : {info['city']}")
print(f"First Name: {info['first_name']}")
print(f"Friends: {info['friends'][3]}")

# ============================================
# Exercise 3: Loops
# ============================================
print("\n=== Exercise 3: Loops ===")

# TODO: Print numbers from 1 to 10
for i in range(1, 11):
    print(i, end=" ")
print()

# Option 2: Print with dash separator (collect all, then print)
print(*range(1, 11), sep="-")

# Option 3: Print with dash separator (using join)
print("-".join(str(i) for i in range(1, 11)))

# TODO: Print each item in the numbers list
for num in numbers:
    print(f"Number: {num}")

# ============================================
# Exercise 4: Functions
# ============================================
print("\n=== Exercise 4: Functions ===")

# TODO: Create a function that calculates the average of a list
def calculate_average(numbers_list):
    """Calculate the average of a list of numbers"""
    if len(numbers_list) == 0:
        return 0
    return sum(numbers_list) / len(numbers_list)

# Test your function
test_numbers = [10, 20, 30, 40, 50]
avg = calculate_average(test_numbers)
print(f"Average of {test_numbers}: {avg}")

# ============================================
# Exercise 5: List Comprehensions
# ============================================
print("\n=== Exercise 5: List Comprehensions ===")

# TODO: Create a list of squares from 1 to 10
squares = [x**2 for x in range(1, 11)]
print(f"Squares: {squares}")

# TODO: Create a list of even numbers from 1 to 20
evens = [x for x in range(1, 21) if x % 2 == 0]
print(f"Even numbers: {evens}")

# ============================================
# Next Steps
# ============================================
print("\n=== Great job! ===")
print("Now move on to:")
print("1. Install NumPy and Pandas: pip install numpy pandas")
print("2. Open 02_data_exploration.ipynb in Jupyter")
print("3. Start learning about data manipulation!")

odds = [x for x in range(1, 21) if x % 2 != 0]
print(f"Odd numbers: {odds}")