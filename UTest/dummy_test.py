from VLLV.Util import Utility as U
from VLLV.Util.Config import Config
from VLLV.Framework.Callbacks import save_batch_image

TEST_STR = ('1.3', '2kb', '3 mb', '4GB', '9Zb', '2.3pB')
ANS = (1.3, 2048.0, 3145728.0, 4294967296.0, 10625324586456701730816.0,
       2589569785738035.2)


def dummy_test_str_to_bytes():
    for t, a in zip(TEST_STR, ANS):
        ans = U.str_to_bytes(t)
        print(t, ans)
        assert ans == a


def dummy_test_config():
    d = Config(a=1, b=2)
    d.update(a=2, b=3)
    d.a = 9
    d.update(Config(b=6, f=5))
    d.pop('b')
    print(d)


def dummy_test_save_batch_image():
    from tensorflow.keras.datasets.cifar10 import load_data
    _, (data, _) = load_data()
    fn = save_batch_image('.', mode='RGB')
    fn(data[:64])


if __name__ == '__main__':
    dummy_test_save_batch_image()
