# Question 3.e

letters_dict = {"B": 0, "K": 1, "O": 2, "-": 3, 0: "B", 1: "K", 2: "O", 3: "-"}
prob_mat = [[0.1, 0.325, 0.25, 0.325],
            [0.4, 0, 0.4, 0.2],
            [0.2, 0.2, 0.2, 0.4],
            [1, 0, 0, 0]]


def find_most_probable_word(k):
    if k == 0:
        return "-"
    words = find_words_for_t(k - 1)
    best_word = max(words, key=lambda x: x[1] * prob_mat[0][letters_dict[x[0][0]]])
    return "B" + best_word[0]


def find_words_for_t(t):
    if t == 0:
        return [("-", 1.0) for _ in range(4)]  # we assume that the word ends with "-"
    previous = find_words_for_t(t-1)
    current = []
    best_prev = max(previous, key=lambda x: x[1])
    best_prev_first_letter = letters_dict[best_prev[0][0]]

    for i in range(3):
        current.append((letters_dict[i] + best_prev[0], best_prev[1] * prob_mat[i][best_prev_first_letter]))
    return current

print(find_most_probable_word(5))
