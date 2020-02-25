import sys
from ner.ner_model import NERModel
from sim.sim_model import SIMModel

def ner():
    pass

def attribute_rank():
    pass

def attribute_match():
    pass

def online_kbqa():
    print('online kbqa start...')
    print('ner test...')
    model_ner = NERModel()
    ans = model_ner.online_predict("我是邺有！你好啊你好啊！")
    print(ans)

    print('sim test...')
    model_sim = SIMModel()
    ans = model_sim.predict("我是邺有！", "邺有")
    print(ans)

def offline_kbqa():
    print('offline kbqa test')

if __name__ == '__main__':
    # mode = sys.argv[1]
    mode = 'online'

    if mode == 'online':
        online_kbqa()
    else:
        offline_kbqa()