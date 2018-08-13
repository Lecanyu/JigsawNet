'''
This file save some common data structures
'''

import operator
import numpy as np

class GtPose:
    def __init__(self, gt_filename):
        self.data = []
        with open(gt_filename) as f:
            all_line = [line.rstrip() for line in f]
            for index, line in enumerate(all_line):
                if index%2!=0:
                    m1, m2, m3, m4, m5, m6, m7, m8, m9 = [t(s) for t, s in zip((float, float, float, float, float, float, float, float, float), line.split())]
                    pose = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])
                    self.data.append(pose)

        # transform ground truth to identity
        baseline = np.linalg.inv(self.data[0])
        for i in range(len(self.data)):
            self.data[i] = np.matmul(baseline, self.data[i])

class GtPoseMatrix:
    def __init__(self, gt_filename):
        self.data=list()
        with open(gt_filename) as f:
            all_line = [line.rstrip() for line in f]
            for i in range(len(all_line)//4):
                matrix = all_line[i*4:(i+1)*4]
                line1 = matrix[0]
                line2 = matrix[1]
                line3 = matrix[2]
                line4 = matrix[3]
                fragmentId = int(line1)
                assert i==fragmentId
                m1, m2, m3 = [t(s) for t, s in zip((float, float, float), line2.split())]
                m4, m5, m6 = [t(s) for t, s in zip((float, float, float), line3.split())]
                m7, m8, m9 = [t(s) for t, s in zip((float, float, float), line4.split())]
                pose = np.array([[m1, m2, m3], [m4, m5, m6], [m7, m8, m9]])
                self.data.append(pose)



class PoseContainer:
    def __init__(self, pose_list):
        self.data = pose_list

    def SaveToFile(self, filename):
        with open(filename, 'w') as f:
            for fragmentId, pose in self.data:
                data_item = "%d\t%d\n%f\t%f\t%f\n%f\t%f\t%f\n%f\t%f\t%f\n"%(fragmentId, fragmentId, pose[0, 0], pose[0, 1], pose[0, 2], pose[1, 0], pose[1, 1], pose[1, 2], pose[2, 0], pose[2, 1], pose[2, 2])
                f.write(data_item)

    def CompareWithGT(self, gtpose, t_threshold, r_threshold):
        evaluation = dict()
        all_successful = True
        for fragmentId, pose in self.data:
            gt_pose = gtpose.data[fragmentId]
            err_mat = np.matmul(pose, np.linalg.inv(gt_pose))

            theta = np.arccos(err_mat[0,0])*180/3.1415926
            t = np.sqrt(err_mat[0,2]**2+err_mat[1,2]**2)
            if theta<r_threshold and t<t_threshold:
                evaluation[fragmentId] = True
            else:
                evaluation[fragmentId] = False
                all_successful = False
        return evaluation, all_successful

class Transform2d:
    def __init__(self, v1=-1, v2=-1, score=-1, transform=np.identity(3), stitchLine=None):
        self.frame1 = v1
        self.frame2 = v2
        self.score = score
        self.transform = transform
        self.stitchLine = stitchLine

        # rank between frame1 and frame2
        self.rank = -1

class Alignment2d:
    def __init__(self, relative_transform_filename):
        self.data = []
        # for example, {'0 1': [0,1,2]} means from 0--1 to find data[0,1,2]
        self.mapIdpair2Transform = {}
        # for example, {'0 1 1': 0} means from 0--1 and 1st to find data[0]
        self.mapIdpairRank2Transform = {}
        # for example, {0: [0,1,2,100]} means from 0 to find data[0,1,2,100] in which either 0-x or x-0
        self.id2Transform = {}

        with open(relative_transform_filename) as f:
            all_line = [line.rstrip() for line in f]
            node_num = 0
            for line in all_line:
                if line[0:4] == "Node":
                    node_num+=1
                else:
                    data_str_list = line.split()
                    v1,v2,score, m1,m2,m3,m4,m5,m6,m7,m8,m9 = [t(s) for t,s in zip((int,int, float, float,float,float,float,float,float,float,float,float), data_str_list[0:12])]
                    transform = np.array([[m1,m2,m3], [m4,m5,m6], [m7,m8,m9]])

                    stitchLine = []
                    stitch_line_c = data_str_list[13:]
                    for i in range(len(stitch_line_c)//2):
                        col = float(stitch_line_c[i*2])
                        row = float(stitch_line_c[i*2+1])
                        stitchLine.append([row, col])
                    self.data.append(Transform2d(v1, v2, score, transform, stitchLine))


        self.data = sorted(self.data, key=operator.attrgetter('score'), reverse=True)
        self.data = sorted(self.data, key=operator.attrgetter('frame2'))
        self.data = sorted(self.data, key=operator.attrgetter('frame1'))

        for i, item in enumerate(self.data):
            idpair = '%d %d'%(item.frame1, item.frame2)
            if idpair in self.mapIdpair2Transform:
                self.mapIdpair2Transform[idpair].append(i)
            else:
                self.mapIdpair2Transform[idpair] = [i]
            if item.frame1 in self.id2Transform:
                self.id2Transform[item.frame1].append(i)
            else:
                self.id2Transform[item.frame1] = [i]
            if item.frame2 in self.id2Transform:
                self.id2Transform[item.frame2].append(i)
            else:
                self.id2Transform[item.frame2] = [i]

        for key,value in self.mapIdpair2Transform.items():
            for rank, index in enumerate(value):
                new_key = "%s %d"%(key, rank+1)
                self.mapIdpairRank2Transform[new_key] = index
                self.data[index].rank = rank+1

def ExpandROI(aligned_img, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col, max_expand_threshold=32):
    min_row1, min_col1, max_row1, max_col1 = bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col
    min_row2, min_col2, max_row2, max_col2 = bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col

    ''' 1. try to move upper and lower boundary first'''
    # upper boundary move up
    for i in range(1, max_expand_threshold):
        row = bbox_min_row-i
        if row<0:
            break
        bg_pixel_count = 0
        for col in range(bbox_min_col, bbox_max_col):
            if np.array_equal(aligned_img[row, col], [0,0,0]):
                bg_pixel_count+=1
        bg_pixel_ratio = bg_pixel_count/(bbox_max_col-bbox_min_col)

        if bg_pixel_count > 5 or bg_pixel_ratio>0.5:
            break
        else:
            min_row1 = row

    # lower boundary move down
    for i in range(1, max_expand_threshold):
        row = bbox_max_row + i
        if row>=aligned_img.shape[0]:
            break
        bg_pixel_count = 0
        for col in range(bbox_min_col, bbox_max_col):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (bbox_max_col - bbox_min_col)

        if bg_pixel_count > 5 or bg_pixel_ratio>0.5:
            break
        else:
            max_row1 = row

    # left boundary move left
    for i in range(1, max_expand_threshold):
        col = bbox_min_col - i
        if col<0:
            break
        bg_pixel_count = 0
        for row in range(min_row1, max_row1):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (max_row1 - min_row1)

        if bg_pixel_count > 5 or bg_pixel_ratio>0.5:
            break
        else:
            min_col1 = col

    # right boundary move right
    for i in range(1, max_expand_threshold):
        col = bbox_max_col + i
        if col>=aligned_img.shape[1]:
            break
        bg_pixel_count = 0
        for row in range(min_row1, max_row1):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (max_row1 - min_row1)

        if bg_pixel_count > 5 or bg_pixel_ratio>0.5:
            break
        else:
            max_col1 = col

    are1 = (max_col1-min_col1)*(max_row1-min_row1)

    ''' 2.  try to move left and right boundary first'''
    # left boundary move left
    for i in range(1, max_expand_threshold):
        col = bbox_min_col - i
        if col < 0:
            break
        bg_pixel_count = 0
        for row in range(bbox_min_row, bbox_max_row):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (bbox_max_row - bbox_min_row)

        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            min_col2 = col

    # right boundary move right
    for i in range(1, max_expand_threshold):
        col = bbox_max_col + i
        if col >= aligned_img.shape[1]:
            break
        bg_pixel_count = 0
        for row in range(bbox_min_row, bbox_max_row):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (bbox_max_row - bbox_min_row)

        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            max_col2 = col
    # upper boundary move up
    for i in range(1, max_expand_threshold):
        row = bbox_min_row - i
        if row < 0:
            break
        bg_pixel_count = 0
        for col in range(min_col2, max_col2):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (max_col2 - min_col2)

        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            min_row2 = row

    # lower boundary move down
    for i in range(1, max_expand_threshold):
        row = bbox_max_row + i
        if row >= aligned_img.shape[0]:
            break
        bg_pixel_count = 0
        for col in range(min_col2, max_col2):
            if np.array_equal(aligned_img[row, col], [0, 0, 0]):
                bg_pixel_count += 1
        bg_pixel_ratio = bg_pixel_count / (max_col2 - min_col2)

        if bg_pixel_count > 5 or bg_pixel_ratio > 0.5:
            break
        else:
            max_row2 = row
    are2 = (max_col2-min_col2)*(max_row2-min_row2)
    if are1>are2:
        return [min_row1, min_col1, max_row1, max_col1]
    else:
        return [min_row2, min_col2, max_row2, max_col2]


def ConvertRawStitchLine2BBoxRatio(raw_stitch_line, stitched_img, transform, offset_transform, max_expand_threshold):
    new_stitch_line = []
    for pt in raw_stitch_line:
        row = pt[0]
        col = pt[1]
        new_pt = np.matmul(transform, np.array([row, col, 1]))
        new_stitch_line.append([new_pt[0], new_pt[1]])
    for i in range(len(new_stitch_line)):
        new_stitch_line[i][0] +=offset_transform[0, 2]
        new_stitch_line[i][1] +=offset_transform[1, 2]
    a = np.transpose(new_stitch_line)
    bbox_min_row = np.floor(np.min(a[0])).astype(int)
    bbox_min_col = np.floor(np.min(a[1])).astype(int)
    bbox_max_row = np.ceil(np.max(a[0])).astype(int)
    bbox_max_col = np.ceil(np.max(a[1])).astype(int)
    [new_min_row, new_min_col, new_max_row, new_max_col] = ExpandROI(stitched_img, bbox_min_row, bbox_min_col, bbox_max_row, bbox_max_col, max_expand_threshold=max_expand_threshold)
    rows, cols, channels = stitched_img.shape
    new_min_row_ratio = new_min_row/rows
    new_min_col_ratio = new_min_col/cols
    new_max_row_ratio = new_max_row/rows
    new_max_col_ratio = new_max_col/cols
    return [new_min_row_ratio, new_min_col_ratio, new_max_row_ratio, new_max_col_ratio]


def calculatePoseErr(gt_pose, pose):
    err = np.matmul(gt_pose, np.linalg.inv(pose))
    if np.abs(err[0, 0] - 1) < 1e-3:
        err[0, 0] = 1
    if np.abs(err[0, 0] + 1) < 1e-3:
        err[0, 0] = -1
    r_err = np.arccos(err[0, 0]) * 180 / np.pi
    t_err = np.sqrt(err[0, 2] ** 2 + err[1, 2] ** 2)

    return [r_err, t_err]