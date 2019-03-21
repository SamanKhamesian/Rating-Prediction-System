from random import sample

import numpy as np

from Source.addresses import RAW_DATA, TRAIN_DATA, VALIDATION_DATA


def make_train_and_test_file():
    with open(RAW_DATA, 'r') as data:
        data_list = data.read().splitlines()
        random_shuffled_data = sample(data_list, len(data_list))
        index = int(len(random_shuffled_data)*0.7)
        training_set, testing_set = random_shuffled_data[:index], random_shuffled_data[index:]

        save_list_to_file(training_set, TRAIN_DATA)
        save_list_to_file(testing_set, VALIDATION_DATA)


def save_list_to_file(input_list, file_name):
    with open(file_name, 'w') as file:
        for line in input_list:
            file.write("%s\n" % line)


def find_max_users_and_items_id():
    with open(RAW_DATA, 'r') as data:
        max_user_id, max_item_id = 0, 0
        for line in data:
            user_id, item_id, score = [int(numbers) for numbers in line.split(',')]
            max_user_id = max(max_user_id, user_id)
            max_item_id = max(max_item_id, item_id)
        return max_user_id + 1, max_item_id + 1


def make_user_item_matrix(n, m):
    with open(TRAIN_DATA, 'r') as data:
        matrix = np.zeros((n, m))
        number_of_scores, sum_of_scores = 0, 0
        for line in data:
            user_id, item_id, score = [int(numbers) for numbers in line.split(',')]
            matrix[user_id, item_id] = score
            sum_of_scores += score
            number_of_scores += 1
        return matrix, sum_of_scores / number_of_scores
