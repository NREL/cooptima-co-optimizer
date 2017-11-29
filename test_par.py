from multiprocessing import Pool
import time



def f(x,y):
    print x*y

if __name__ == '__main__':
    pool = Pool(processes=4)

    pool.map(f, [[1,2],[2,1]])
    r  = pool.map_async(f, [[1,2],[2,1]])
    # DO STUFF
    print 'HERE'
    print 'MORE'
    r.wait()
    print 'DONE'