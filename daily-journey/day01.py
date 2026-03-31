import lib.list_utils as utils
import lib.time_utils as time_utils
import random

max_value = 500
list_size = 20
repeat_count = 100000
threshold = 50

listh = ["apple",  "banana", "cherry"]
print(listh)

selected_item = random.choice(listh)


print("Selected item:", selected_item)



# Benchmark: randint loop vs random.sample over 100k iterations
avg1 = time_utils.average_time(utils.random_number_1, max_value, list_size, repeat=repeat_count)
print(f"random_number_1: {avg1:.7f}s")

avg2 = time_utils.average_time(utils.random_number_2, max_value, list_size, repeat=repeat_count)
print(f"random_number_2: {avg2:.7f}s")

print("compare avg times:", avg1 > avg2)

# Same list used for both filters to ensure fair comparison
rand_list = utils.random_number_1(max_value, list_size)
numbers1 = utils.filtered_numbers_1(rand_list, threshold)
print("Filtered numbers1:", numbers1)

numbers2 = utils.filtered_numbers_2(rand_list, threshold)
print("Filtered numbers2:", numbers2)