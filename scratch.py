import random



def generateRandomBatch(input, batch_size):
    random.shuffle(input)
    for i in range(0, len(input), batch_size):
        yield input[i:i + batch_size]


a = list(range(10))

b = generateRandomBatch(a, 3)


