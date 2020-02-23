import sys
from ner.ner_model import NERModel

def ner():
    pass

def attribute_rank():
    pass

def attribute_match():
    pass

def online_kbqa():
    print('offline kbqa start...')
    model_ner = NERModel()
    ans = model_ner.online_predict("我是邺有，你好啊你好啊！")
    print(ans)

def offline_kbqa():
    print('offline kbqa test')

if __name__ == '__main__':
    mode = sys.argv[1]

    if mode == 'online':
        online_kbqa()
    else:
        offline_kbqa()