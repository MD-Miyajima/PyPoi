import time

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
    file_paths = [f for f in sorted(glob("{}/*".format(path)), key=natural_keys) if
                  re.search(pattern, f, re.IGNORECASE)]

    return file_paths


# 画像ファイルの存在確認
def exist_check(img_dir='./testimages'):
    # 戻り値
    result = False

    # 画像の存在確認
    target_dir = img_dir + '/target'
    source_path = img_dir + '/source'
    source_dir = source_path + '/source'
    groundtruth_dir = source_path + '/groundtruth'
    original_dir = source_path + '/original'
    groundtruth_crop_dir = source_path + '/groundtruth_crop'
    original_crop_dir = source_path + '/original_crop'
    mask_dir = source_path + '/mask'
    ret_dir = img_dir + '/ret'
    targets = img_reader(target_dir)
    originals = img_reader(original_dir)

    # img_dirの存在確認
    if exists(img_dir):

        # targets, sourcesの存在確認
        if len(targets) != 0 and len(originals) != 0:
            print('必要なディレクトリと画像ファイルの存在を確認しました。')
            print('targets:{}'.format(len(targets)))
            print('sources:{}'.format(len(originals)))

            # /mask、/ret, /source, /original, /groundtruth が存在していたら消す。
            if exists(mask_dir):
                rmtree(mask_dir)
            if exists(ret_dir):
                rmtree(ret_dir)
            if exists(source_dir):
                rmtree(source_dir)
            if exists(original_crop_dir):
                rmtree(original_crop_dir)
            if exists(groundtruth_crop_dir):
                rmtree(groundtruth_crop_dir)

            # /mask, /ret ディレクトリを作成
            makedirs(mask_dir)
            makedirs(ret_dir)
            makedirs(source_dir)
            makedirs(original_crop_dir)
            makedirs(groundtruth_crop_dir)

            result = True

        else:
            print('"./target/"に合成先画像を、\n'
                  '"./source/original/"に合成元画像を、\n'
                  '"./source/groundtruth/"に合成元画像のGroundTruthを、\n'
                  '作成してください。')
            makedirs(target_dir, exist_ok=True)
            makedirs(groundtruth_dir, exist_ok=True)
            makedirs(original_dir, exist_ok=True)

    return result


# モルフォロジー変換
def erosion(img):
    kernel = np.ones((3, 3), np.uint8)
    cvt_erosion = cv2.erode(img, kernel, iterations=1)
    return cvt_erosion


# モルフォロジー変換
def dilation(img):
    kernel = np.ones((3, 3), np.uint8)
    cvt_dilation = cv2.dilate(img, kernel, iterations=2)
    return cvt_dilation


# 入力画像(RGB)の中央を正方形にクロップした画像を出力する関数
def crop_square(img):
    h, w = img.shape
    x = w if h > w else h
    y = x

    top = int((h - y) / 2)
    bottom = top + y
    left = int((w - x) / 2)
    right = left + x

    img = img[top:bottom, left:right]
    return img


# テンプレートマッチング
def template_matching(img, template):
    # 元画像の高さ・幅
    ih, iw = img.shape
    # テンプレート画像の高さ・幅
    h, w = template.shape

    # テンプレートマッチング（OpenCVで実装）
    match = cv2.matchTemplate(img, template, cv2.TM_SQDIFF_NORMED)
    min_value, max_value, min_pt, max_pt = cv2.minMaxLoc(match)
    pt = min_pt

    # mask画像を作成（imgと同じ画像サイズ）
    sw = pt[0]
    sh = pt[1]
    ew = pt[0] + w
    eh = pt[1] + h
    mask = np.zeros((ih, iw), np.uint8)
    mask[sh:eh, sw:ew] = template

    # マスク
    img[mask == 0] = 0

    return img


# 画像検査機用のポアソン合成
class poissonBlending:

    # コンストラクタ
    def __init__(self, img_dir: str):
        self.img_dir = img_dir
        self.source_dir = self.img_dir + '/source'
        self.targets = img_reader(self.img_dir + '/target')
        self.sources = img_reader(self.source_dir + '/source')
        self.masks = img_reader(self.source_dir + '/mask')
        self.original = img_reader(self.source_dir + '/original')
        self.groundtruth = img_reader(self.source_dir + '/groundtruth')
        self.ret = self.img_dir + '/ret'

    # テンプレートマッチング
    def template_matching(self, x=512, y=512):
        template = cv2.imread(self.source_dir + "/template.png")
        template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        originals = img_reader(self.source_dir + '/original')
        groundtruths = img_reader(self.source_dir + '/groundtruth')
        resize = (x, y)
        for i, (original, groundtruth) in enumerate(zip(originals, groundtruths)):
            # 画像の読み込み
            img_org = cv2.imread(original)
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            img_grt = cv2.imread(groundtruth)
            img_grt = cv2.cvtColor(img_grt, cv2.COLOR_BGR2GRAY)

            # original画像に対する処理
            out_org = template_matching(img_org, template)
            crop_org = crop_square(out_org)
            resize_org = cv2.resize(crop_org, resize)

            # groundtruthの画像に対する処理
            crop_grt = crop_square(img_grt)
            resize_grt = cv2.resize(crop_grt, resize)
            resize_grt[resize_grt != 0] = 255

            # 画像の保存
            cv2.imwrite(self.source_dir + '/original_crop/original_crop_{}.bmp'.format(i), resize_org)
            cv2.imwrite(self.source_dir + '/groundtruth_crop/groundtruth_crop_{}.bmp'.format(i), resize_grt)

    # source画像の作成
    def make_source(self):
        originals = img_reader(self.source_dir + '/original_crop')
        groundtruths = img_reader(self.source_dir + '/groundtruth_crop')
        for i, (original, groundtruth) in enumerate(zip(originals, groundtruths)):

            # 画像の読み込み
            img_org = cv2.imread(original)
            img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
            img_grt = cv2.imread(groundtruth)
            img_grt = cv2.cvtColor(img_grt, cv2.COLOR_BGR2GRAY)
            img_grt = dilation(img_grt)

            # groundtruthのマスク領域を抽出
            rows = np.any(img_grt, axis=1)
            cols = np.any(img_grt, axis=0)
            y_min, y_max = np.where(rows)[0][[0, -1]]
            x_min, x_max = np.where(cols)[0][[0, -1]]
            crop_grt = img_grt[y_min - 1:y_max + 2, x_min - 1:x_max + 2]
            crop_org = img_org[y_min - 1:y_max + 2, x_min - 1:x_max + 2]
            crop_org[crop_grt == 0] = 0
            cv2.imwrite(self.source_dir + '/source/source_{}.bmp'.format(i), crop_org)

    # source画像からmask画像を作成
    def make_mask(self):
        sources = img_reader(self.source_dir + '/source')
        for i, source in enumerate(sources):
            np_mask = np.array(PIL.Image.open(source))
            np_mask[np_mask != 0] = 255
            np_mask = erosion(np_mask)
            mask = PIL.Image.fromarray(np.uint8(np_mask))
            mask.save(self.source_dir + '/mask/mask_{}.bmp'.format(i))

    # AI用ポアソン合成
    def blend(self):

        index = 0

        # ターゲット取得
        for i, target in enumerate(self.targets):
            # 合成先画像の読み込み（targetsより順番に選択）
            np_target = np.array(PIL.Image.open(target).convert('RGB'))

            # 合成元画像の読み込み（sources画像に偏りがないように選択）
            # [20210617の突貫的なもの] target画像100枚に対してsource画像25枚なので、target画像4枚に対して1つのsource画像が使われるようにしている。
            index = int(i/4)
            src_img = PIL.Image.open(self.sources[index]).convert('RGB')
            np_source = np.array(src_img)

            # マスク画像の読み込み（sourceと同名のマスク画像を選択）
            np_mask = np.array(PIL.Image.open(self.masks[index]).convert('L'))

            # マスク領域外の座標をランダムに選択
            taples = list(zip(*np.where(np_target[:, :, 0] != 0)))
            offset = taples[randint(0, len(taples))]

            # ポアソン合成
            img_ret = blend(np_target, np_source, np_mask, offset=offset)
            img_ret = PIL.Image.fromarray(np.uint8(img_ret))
            img_ret.save(self.ret + '/ret__{}.bmp'.format(i))


# メイン関数
def main():

    # 画像のディレクトリ
    img_dir = './testimages'

    print('ポアソン合成 for AI画像検査機 を 開始します...')
    poi = poissonBlending(img_dir)

    # 画像が適切に存在していた場合
    if exist_check(img_dir):
        poi.template_matching()
        poi.make_source()
        poi.make_mask()
        poi.blend()


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print(end-start)
