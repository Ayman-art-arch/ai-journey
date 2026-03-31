import random

# Allows duplicates — each call to randint is independent
def random_number_1(x, y):
    return [random.randint(1, x) for i in range(1, y+1)]

# No duplicates — samples unique values from range
def random_number_2(x, y):
    return  random.sample(range(1, x+1), y)

# Functional style: filter() + lambda
def filtered_numbers_1(numbers, threshold):
    return list(filter(lambda x: x > threshold, numbers))

# Pythonic style: list comprehension (preferred in most cases)
def filtered_numbers_2(numbers, threshold):
    return [x for x in numbers if x > threshold]