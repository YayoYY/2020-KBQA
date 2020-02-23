import sys
from run_ner import NERModel

def ner():
    pass

def attribute_rank():
    pass

def attribute_match():
    pass

def online_kbqa():
    print('online kbqa test')
    ner = NERModel()
    ans = ner.predict("我是尤玥")
    print(ans)

def offline_kbqa():
    print('offline kbqa test')

if __name__ == '__main__':
    mode = 'online'
    if mode == 'online':
        online_kbqa()

    else:
        offline_kbqa()

