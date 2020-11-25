import random
def get_label():
    sampleList = [0, 0, 0, 0, 0, 0, 0, 1, 1, 2] #: 0 train 1 validate 2 test
    x = random.choice(sampleList)
    if x == 0:
        return 'train'
    elif x == 1:
        return 'val'
    else:
        return 'test'
label = get_label()
print('label', label)