import multiprocessing

def worker(input_num):
    #worker function
    print ('Worker')
    x = 0
    while x < 10000000000000000:
        print(x)
        x += 1
    return

if __name__ == '__main__':
    jobs = []
    for i in range(50):
        p = multiprocessing.Process(target=worker, args=(3))
        jobs.append(p)
        p.start()