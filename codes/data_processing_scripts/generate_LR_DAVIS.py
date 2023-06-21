import os
import sys
import cv2
import numpy as np
import os.path as osp
import glob

try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.util import imresize_np
except ImportError:
    pass


def generate_LR_DAVIS():
    # set parameters
    up_scale = 4
    mod_scale = 4
    resolution = '1280x720' # '1280x720' or '1920x1080'
    # set data dir
    sourcedir = '../../datasets/DAVIS-2019/GT/{}'.format(resolution)
    saveLRpath = '../../datasets/DAVIS-2019/BIx4/{}'.format(resolution)
    txt_file_path = '../../datasets/DAVIS-2019/GT/{}'.format(resolution)
    txt_file_list = sorted(glob.glob(osp.join(txt_file_path, '*.txt')))

    for txt_file in txt_file_list:
        #### read all the image paths to a list
        print('Reading image path list ...')
        with open(txt_file) as f:
            train_l = f.readlines()
            train_l = [v.strip() for v in train_l]
        all_img_list = []
        for line in train_l:
            folder = line.split('/')[0]
            seq = line.split('/')[1]
            all_img_list.extend(glob.glob(osp.join(sourcedir, folder, seq)))
        all_img_list = sorted(all_img_list)
        num_files = len(all_img_list)

        # prepare data with augementation
        for i in range(num_files):
            filename = all_img_list[i]
            print('No.{} -- Processing {}'.format(i, filename))

            # read image
            image = cv2.imread(filename)

            width = int(np.floor(image.shape[1] / mod_scale))
            height = int(np.floor(image.shape[0] / mod_scale))

            # modcrop
            if len(image.shape) == 3:
                image_HR = image[0:mod_scale * height, 0:mod_scale * width, :]
            else:
                image_HR = image[0:mod_scale * height, 0:mod_scale * width]

            # LR
            image_LR = imresize_np(image_HR, 1 / up_scale, True)

            folder = filename.split('/')[7]
            name = filename.split('/')[8]

            if not os.path.isdir(osp.join(saveLRpath, folder)):
                os.mkdir(osp.join(saveLRpath, folder))

            cv2.imwrite(osp.join(saveLRpath, folder, name), image_LR)

    print('Finish LR generation')


if __name__ == "__main__":
    generate_LR_DAVIS()
