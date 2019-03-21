from math import sqrt

from Source.config import *
from Source.utilities import *
from Source.addresses import VALIDATION_DATA


def pre_process(mu, value, user_id, item_id, beta_user, beta_item, user_features, item_features):

    # calculate prediction error for gradient (use explicitly model bias)
    error = value - (mu + beta_user[user_id] + beta_item[item_id] + np.dot(user_features[:, user_id].T, item_features[:, item_id]))

    # modify parameters in the opposite direction of the gradient
    beta_user[user_id] += GAMMA * (error - LAMBDA * beta_user[user_id])
    beta_item[item_id] += GAMMA * (error - LAMBDA * beta_item[item_id])

    # update latent user and item features vector
    user_features[:, user_id] += GAMMA * (error * item_features[:, item_id] - LAMBDA * user_features[:, user_id])
    item_features[:, item_id] += GAMMA * (error * user_features[:, user_id] - LAMBDA * item_features[:, item_id])


def calculate_rmse(list_of_scores):
    n = len(list_of_scores)
    sum = 0
    for x in list_of_scores:
        sum += ((x[0] - x[1])**2)
    average = sum / n
    result = sqrt(average)
    return result


def predict_scores():
    best_rmse = MAX_RMSE
    n, m = find_max_users_and_items_id()

    # mu is global bias - average of all the train Data labels
    matrix, mu = make_user_item_matrix(n, m)
    users, items = matrix.nonzero()

    # user and item bias
    beta_user = (MUL * np.zeros((n, 1))).flatten()
    beta_item = (MUL * np.zeros((m, 1))).flatten()

    # create user and item features vector
    user_features = MUL * np.random.randn(K, n)
    item_features = MUL * np.random.randn(K, m)

    index = 1
    for epoch in range(TIMES):
        for user_id, item_id in zip(users, items):
            value = matrix[user_id, item_id]
            pre_process(mu, value, user_id, item_id, beta_user, beta_item, user_features, item_features)

        # read validation Data line bt line
        with open(VALIDATION_DATA, 'r') as data:
            result = []
            for line in data:
                user, item, real_score = [int(numbers) for numbers in line.split(',')]
                rate = min(MAX_SCORE, mu + beta_user[user] + beta_item[item] + np.dot(user_features[:, user].T, item_features[:, item]))
                result.append([rate, real_score])

            rmse = calculate_rmse(result)
            print(f'{index:02} : RMSE is {rmse}')
            index += 1

            if rmse > best_rmse and epoch >= LIMIT:
                break
            best_rmse = min(rmse, best_rmse)

    print('--------------------------')
    print('BEST RMSE is : ', best_rmse)


def main():
    make_train_and_test_file()
    predict_scores()


if __name__ == '__main__':
    main()
