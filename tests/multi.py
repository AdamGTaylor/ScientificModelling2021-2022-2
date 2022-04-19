import timeit
import numpy as np
from multiprocessing import Process, Queue, Pool

def calcSum(start,end):
    """
    calculates the sum for start to end of integers
    
    INPUT:
        start: int
        end: int
        
    OUTPUT:
        summa: summed from start to end
    """
    summa = 0
    for i in np.arange(start,end):
        summa+=i
        
    return summa

class Multiprocessor():

    def __init__(self):
        self.processes = []
        self.queue = Queue()

    @staticmethod
    def _wrapper(func, queue, args, kwargs):
        ret = func(*args, **kwargs)
        queue.put(ret)

    def run(self, func, *args, **kwargs):
        args2 = [func, self.queue, args, kwargs]
        p = Process(target=self._wrapper, args=args2)
        self.processes.append(p)
        p.start()

    def wait(self):
        rets = []
        for p in self.processes:
            ret = self.queue.get()
            rets.append(ret)
        for p in self.processes:
            p.join()
        return rets

# tester
if __name__ == "__main__":
    answers = []
    start = timeit.default_timer()
    for i in range(4):
        answers.append(calcSum(10,2e7))
    stop = timeit.default_timer()
    print('Time: ', stop - start) 
    
    pool = Pool(4)

    start = timeit.default_timer()
    r1 = pool.apply_async(calcSum, [10,2e7])
    r2 = pool.apply_async(calcSum, [10,2e7])
    r3 = pool.apply_async(calcSum, [10,2e7])
    r4 = pool.apply_async(calcSum, [10,2e7])

    ans1 = r1.get(timeout=10)
    ans2 = r2.get(timeout=10)
    ans3 = r3.get(timeout=10)
    ans4 = r4.get(timeout=10)
    stop = timeit.default_timer()
    
    print('Time: ', stop - start) 