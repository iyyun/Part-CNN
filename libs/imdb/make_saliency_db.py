import cv2 as cv
import random
import numpy as np
from det_imdb import get_dbinfo
import os, sys
import glob

folder = ['USA', 'ETH', 'INRIA', 'TudBrussels']

class tData:
    def __init__(self, frame=-1, obj_type="unset", occlusion=-1, \
                 obs_angle=-10, x1=-1, y1=-1, x2=-1, y2=-1, track_id=-1):
        """
            Constructor, initializes the object given the parameters.
        """

        # init object data
        self.img_root = 'none'
        self.sal_root = 'none'
        self.frame = frame
        self.track_id = track_id
        self.obj_type = obj_type
        self.occlusion = occlusion
        self.obs_angle = obs_angle
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.vx1 = 0
        self.vy1 = 0
        self.vx2 = 0
        self.vy2 = 0
        self.skip_p = False


mouse_flag = -1
select_rect = [0, 0, 0, 0]

def onMouse(event, x, y, flags, param):
    global select_rect, mouse_flag

    if mouse_flag == 0:
        select_rect[2] = select_rect[0] + np.abs(x - select_rect[0])
        select_rect[3] = select_rect[1] + np.abs(y - select_rect[1])

    if event == cv.EVENT_LBUTTONDOWN:

        if mouse_flag == -1:
            select_rect[0] = x
            select_rect[1] = y
        mouse_flag = 0

    elif event == cv.EVENT_LBUTTONUP:
        if select_rect[2] > 10 and select_rect[3] > 10:
            mouse_flag = 1
        else:
            mouse_flag = -1


# read data
def load_label_file(f_path, mode, dset):
    global select_rect, mouse_flag
    labels = []

    for DB in dset:
        s_f, v_f, start, end = get_dbinfo(folder.index(DB), mode)

        for s in s_f[start:end]:
            idx = s_f.index(s)

            pre_d = '%s/data-%s/saliency' % (f_path, DB)
            if not os.path.exists(pre_d):
                os.mkdir(pre_d)

            s_sal_p = '%s/data-%s/saliency/%s' % (f_path, DB, s)
            if not os.path.exists(s_sal_p):
                os.mkdir(s_sal_p)

            for v in range(0, v_f[idx]):
                v_sal_p = '%s/data-%s/saliency/%s/V%03d' % (f_path, DB, s, v)
                if not os.path.exists(v_sal_p):
                    os.mkdir(v_sal_p)

                path = os.path.join(f_path, 'data-%s/images/%s/V%03d' % (DB, s, v), '*g')

                img_files = glob.glob(path)
                img_files = np.sort(img_files)

                # read
                for fi in img_files:
                    fn = fi.split('/')[-1].split('.')[0]
                    # label data reas
                    label_path = os.path.join(f_path, 'data-%s/annotations/%s/V%03d/%s.txt' % (DB, s, v, fn))

                    label_file = open(label_path, 'r')
                    data_list = []

                    img = cv.imread(fi)

                    use_onMouse = False

                    for line in label_file:
                        line_s = line.strip().split(' ')
                        if line_s[0] == '%':
                            continue

                        t_data = tData()

                        t_data.frame = fn
                        t_data.img_root = fi
                        t_data.sal_root = '%s/data-%s/saliency/%s/V%03d/%s' % (f_path, DB, s, v, fn)
                        t_data.obj_type = line_s[0]

                        #continue

                        t_data.x1 = int(line_s[1])
                        t_data.y1 = int(line_s[2])
                        t_data.x2 = int(line_s[1]) + int(line_s[3])
                        t_data.y2 = int(line_s[2]) + int(line_s[4])

                        if t_data.obj_type == 'people':
                            use_onMouse = True
                            t_data.skip_p = True

                        use_onMouse = True


                        if (t_data.y2 - t_data.y1) < 25:
                            #use_onMouse = True
                            t_data.skip_p = True
                            #continue

                        t_data.occlusion = int(line_s[5])

                        if t_data.occlusion == 1:

                            t_data.vx1 = int(line_s[6])
                            t_data.vy1 = int(line_s[7])
                            t_data.vx2 = int(line_s[6]) + int(line_s[8])
                            t_data.vy2 = int(line_s[7]) + int(line_s[9])


                            if (t_data.vy2 - t_data.vy1) < 15:
                                #use_onMouse = True
                                t_data.skip_p = True
                                #continue
                            if (t_data.vx2 - t_data.vx1) < 15:
                                #use_onMouse = True
                                t_data.skip_p = True
                                #continue


                        data_list.append(t_data)

                    # read data?
                    sal_root = '%s/data-%s/saliency/%s/V%03d/%s.jpg' % (f_path, DB, s, v, fn)
                    if not os.path.exists(sal_root):
                        sal_img = np.zeros(img.shape, np.uint8)
                    else:
                        sal_img = cv.imread(sal_root)

                    for xx in data_list:
                        if xx.skip_p:
                            cv.rectangle(img, (xx.x1, xx.y1), (xx.x2, xx.y2), (0, 0, 255), 1)
                        else:
                            cv.rectangle(img, (xx.x1, xx.y1), (xx.x2, xx.y2), (0, 255, 0), 1)

                        if xx.skip_p == False:
                            if xx.occlusion:
                                cv.rectangle(sal_img, (xx.vx1, xx.vy1), (xx.vx2, xx.vy2), (255, 255, 255), -1)
                            else:
                                cv.rectangle(sal_img, (xx.x1, xx.y1), (xx.x2, xx.y2), (255, 255, 255), -1)

                    ##
                    if use_onMouse:
                        while True:
                            tmp_image = np.copy(img)
                            tmp_image_sal = np.copy(sal_img)
                            if mouse_flag == 0:
                                cv.rectangle(tmp_image, (select_rect[0], select_rect[1]), (select_rect[2], select_rect[3]), (0, 255, 255), 1)

                            if mouse_flag == 1:
                                cv.rectangle(img, (select_rect[0], select_rect[1]), (select_rect[2], select_rect[3]), (0, 255, 255), 1)
                                cv.rectangle(sal_img, (select_rect[0], select_rect[1]), (select_rect[2], select_rect[3]), (255, 255, 255), -1)
                                mouse_flag = -1

                            # temp
                            tmp_image = tmp_image.astype(np.float)
                            tmp_image_sal = tmp_image_sal.astype(np.float)

                            tmp_image = tmp_image * 1./255.
                            tmp_image_sal = tmp_image_sal * 1. / 255.

                            cv.imshow('name', tmp_image*0.6 + tmp_image_sal*0.4)
                            cv.imshow('sal_img', sal_img)
                            key = cv.waitKey(10)
                            if key == 27:
                                break


                    # image save

                    cv.imwrite(sal_root, sal_img)
                    cv.imshow('name', img)
                    cv.imshow('sal_img', sal_img)
                    cv.waitKey(10)



                    label_file.close()
# test

cv.namedWindow('name')
cv.setMouseCallback('name', onMouse)
# folder = ['USA', 'ETH', 'INRIA', 'TudBrussels']
load_label_file('/media/inyong/data/dataset/pedestrian-detection', 'test', ['INRIA'])



