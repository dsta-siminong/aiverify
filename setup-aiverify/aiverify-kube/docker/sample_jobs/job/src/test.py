import os

print("hello world")

var_one_val = os.environ.get('var_one', "not found")
print(f"Value of VAR_ONE: {var_one_val}")

var_two_val = os.environ.get('var_two', "not found")
print(f"Value of VAR_TWO: {var_two_val}")