from poissonblending import blend
import numpy as np
import PIL.Image
from glob import glob
import re
from os import makedirs
from os.path import exists
from random import randint
from shutil import rmtree, copytree
import cv2


# 名前順ソート呪文1
def atoi(text):
    return int(text) if text.isdigit() else text


# 名前順ソート呪文2
def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text)]


# PIL -> OpenCV
def pil2cv(image):
    ''' PIL型 -> OpenCV型 '''
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image


# OpenCV -> PIL
def cv2pil(image):
    ''' OpenCV型 -> PIL型 '''
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
    new_image = PIL.Image.fromarray(new_image)
    return new_image


# 画像のパスをリストにして返す
def img_reader(path: str):
    # 画像の入ったフォルダの画像名とパスを取得
    pattern = ".*\.(jpg|png|bmp)"
    file_paths = [f for f in sorted(glob("{}/*".format(path)), key=natural_keys) if re.search(pattern, f, re.IGNORECASE)]

    return file_paths


# 画像ファイルの存在確認
def exist_check(img_dir='./testimages'):
    # 戻り値
    result = False

    # 画像の存在確認
    target_dir = img_dir + '/target'
    source_dir = img_dir + '/source'
    mask_dir = img_dir + '/mask'
    ret_dir = img_dir + '/ret'
    targets = img_reader(target_dir)
    sources = img_reader(source_dir)

    # img_dirの存在確認
    if exists(img_dir):

        # targets, sourcesの存在確認
        if len(targets) != 0 and len(sources) != 0:
            print('必要なディレクトリと画像ファイルの存在を確認しました。')
            print('targets:{}'.format(len(targets)))
            print('sources:{}'.format(len(sources)))

            # /mask、/ret が存在していたら消す。
            if exists(mask_dir):
                rmtree(mask_dir)
            if exists(ret_dir):
                rmtree(ret_dir)

            # /mask, /ret ディレクトリを作成
            makedirs(mask_dir)
            makedirs(ret_dir)

            result = True

        else:
            print('"./target/"に合成先画像を、\n"./source/"に合成元画像を、作成してください。')
            makedirs(target_dir, exist_ok=True)
            makedirs(source_dir, exist_ok=True)

    return result


# モルフォロジー変換
def erosion(img):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    return erosion

# 画像検査機用のポアソン合成
class poissonBlending:

    # コンストラクタ
    def __init__(self, img_dir: str):
        self.img_dir = img_dir
        self.targets = img_reader(self.img_dir + '/target')
        self.sources = img_reader(self.img_dir + '/source')
        self.masks = img_reader(self.img_dir + '/mask')
        self.ret = self.img_dir + '/ret'

    # source画像からmask画像を作成
    def make_mask(self):
        sources = img_reader(self.img_dir + '/source')
        for i, source in enumerate(sources):
            np_mask = np.array(PIL.Image.open(source))
            np_mask[np_mask != 0] = 255
            np_mask = erosion(np_mask)
            mask = PIL.Image.fromarray(np.uint8(np_mask))
            mask.save(self.img_dir + '/mask/mask_{}.bmp'.format(i))

    # AI用ポアソン合成
    def blend(self):

        # ターゲット取得
        for i, target in enumerate(self.targets):
            # 合成先画像の読み込み（targetsより順番に選択）
            np_target = np.array(PIL.Image.open(target))

            # 合成元画像の読み込み（sourcesよりランダムに選択）
            index = randint(0, len(self.sources) - 1)
            np_source = np.array(PIL.Image.open(self.sources[index]).convert('RGB'))

            # マスク画像の読み込み（sourceと同名のマスク画像を選択）
            np_mask = np.array(PIL.Image.open(self.masks[index]).convert('L'))

            # 合成する箇所をランダム化
            offset_y = randint(100, 300)
            offset_x = randint(100, 300)

            # ポアソン合成
            img_ret = blend(np_target, np_source, np_mask, offset=(offset_y, offset_x))
            img_ret = PIL.Image.fromarray(np.uint8(img_ret))
            img_ret.save(self.ret + '/ret_{}.bmp'.format(i))

# メイン関数
def main():

    # 画像のディレクトリ
    img_dir = './testimages'

    print('ポアソン合成 for AI画像検査機 を 開始します...')
    poi = poissonBlending(img_dir)

    # 画像が適切に存在していた場合
    if exist_check(img_dir):
        poi.make_mask()
        poi.blend()


if __name__ == "__main__":
    main()

