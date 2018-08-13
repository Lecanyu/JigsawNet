'''
Convert pairwise alignment transformation to one stitched image
'''

import cv2
import numpy as np

'''
transform dst to src
'''
def FusionImage(src, dst, transform, bg_color=[0,0,0]):
    black_bg = [0,0,0]
    if bg_color!=black_bg:
        src[np.where((src == bg_color).all(axis=2))] = [0,0,0]
        dst[np.where((dst == bg_color).all(axis=2))] = [0, 0, 0]

    color_indices = np.where((dst != black_bg).any(axis=2))
    color_pt_num = len(color_indices[0])
    one = np.ones(color_pt_num)

    color_indices = list(color_indices)
    color_indices.append(one)
    color_indices = np.array(color_indices)

    transformed_lin_pts = np.matmul(transform, color_indices)
    # bounding box after transform
    try:
        dst_min_row = np.floor(np.min(transformed_lin_pts[0])).astype(int)
        dst_min_col = np.floor(np.min(transformed_lin_pts[1])).astype(int)
        dst_max_row = np.ceil(np.max(transformed_lin_pts[0])).astype(int)
        dst_max_col = np.ceil(np.max(transformed_lin_pts[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    # global bounding box
    src_color_indices = np.where((src != black_bg).any(axis=2))
    try:
        src_min_row = np.floor(np.min(src_color_indices[0])).astype(int)
        src_min_col = np.floor(np.min(src_color_indices[1])).astype(int)
        src_max_row = np.ceil(np.max(src_color_indices[0])).astype(int)
        src_max_col = np.ceil(np.max(src_color_indices[1])).astype(int)
    except ValueError:
        return []       # the src or dst image has the same color with background. e.g totally black.

    min_row = min(dst_min_row, src_min_row)
    max_row = max(dst_max_row, src_max_row)
    min_col = min(dst_min_col, src_min_col)
    max_col = max(dst_max_col, src_max_col)

    offset_row = -min_row
    offset_col = -min_col

    offset_transform = np.float32([[1,0,offset_col],[0,1,offset_row]])
    dst_transform = np.matmul(np.matrix([[1,0,offset_row],[0,1,offset_col],[0,0,1]]), transform)
    # convert row, col to opencv x,y
    dst_transform = np.float32([[dst_transform[0,0], dst_transform[1,0], dst_transform[1,2]], [dst_transform[0,1], dst_transform[1,1], dst_transform[0,2]]])

    src_transformed = cv2.warpAffine(src, offset_transform, (max_col-min_col, max_row-min_row))
    dst_transformed = cv2.warpAffine(dst, dst_transform, (max_col-min_col, max_row-min_row))

    # overlap detection
    a = np.all(src_transformed == black_bg, axis=2)
    b = np.all(dst_transformed != black_bg, axis=2)
    c = np.logical_and(a, b)
    c = c.reshape((c.shape[0], c.shape[1]))
    non_overlap_indices = np.where(c)
    if len(np.where(b)[0]) == 0:
        assert False and "no valid pixels in transformed dst image, please check the transform process"
    else:
        overlap_ratio = 1 - len(non_overlap_indices[0]) / len(np.where(b)[0])

    # fusion
    bg_indices = np.where(a)
    src_transformed[bg_indices] = dst_transformed[bg_indices]

    offset_transform_matrix = np.float32([[1, 0, offset_row], [0, 1, offset_col], [0,0,1]])
    return [src_transformed, overlap_ratio, offset_transform_matrix]