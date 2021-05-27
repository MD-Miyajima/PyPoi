import PIL
import numpy as np
from poissonblending import blend
from os import makedirs
from glob import glob
import random

def test():

    # 画像の存在確認
    img_dir = './testimages'
    target_dir = img_dir + '/target'
    source_dir = img_dir + '/source'
    targets = glob(target_dir + '/*.png')
    sources = glob(source_dir + '/*.png')
    if len(targets) != 0 and len(sources) != 0:
        print('必要なディレクトリと画像ファイルの存在を確認しました。')
        print('targets:{}'.format(len(targets)))
        print('sources:{}'.format(len(sources)))
    else:
        print('"./target/"に合成先画像を、\n"./source/"に合成元画像を、作成してください。')
        makedirs(target_dir, exist_ok=True)
        makedirs(source_dir, exist_ok=True)

    for target in targets:

        # 合成先画像の読み込み（targetsより順番に選択）
        img_target = np.array(PIL.Image.open(target))

        # 合成元画像の読み込み（sourcesよりランダムに選択）
        source = random.choice(sources)
        img_source = np.array(PIL.Image.open(source))

        # マスク画像の作成（合成元画像より作成）
        # img_mask = (def)

        img_ret = blend(img_target, img_source, img_mask, offset=(40, -100))
        img_ret = PIL.Image.fromarray(np.uint8(img_ret))
        img_ret.save('./testimages/test1_ret.png')


if __name__ == '__main__':
    test()