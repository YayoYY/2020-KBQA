import sys

def ner():
    pass

def attribute_rank():
    pass

def attribute_match():
    pass

def online_kbqa():
    print('online kbqa test')
    pass

def offline_kbqa():
    print('offline kbqa test')
    pass

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'online':
        online_kbqa()
    else:
        offline_kbqa()