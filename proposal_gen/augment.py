import cv2
import numpy as np
import random

def random_flip(image, masks):
    image = cv2.flip(image, 1)
    ret_masks = []
    for i in range(masks.shape[2]):
        mask = masks[:, :, i]
        ret_masks.append( cv2.flip(mask, 1) )
    masks = np.asarray(ret_masks)
    masks = np.transpose(masks, (1, 2, 0))
    return image, masks


def random_crop(image, masks, class_ids, coe=0.7):
    h, w = image.shape[:2]
    nh, nw = ( int(h * coe), int(w * coe) )
    random_x = random.choice(  range( 0, w - nw )   )
    random_y = random.choice( range( 0, h - nh )  )

    ret_image = image[random_y:(random_y+nh), random_x:(random_x+nw), :]

    ret_masks = []
    drop_index = []
    for i in range(masks.shape[2]):
        mask = masks[:,:,i]
        mask = mask[random_y:(random_y+nh), random_x:(random_x + nw)]

        # ignore mask which has a height or width less than 3 pixels
        mask_height = np.sum(np.any(mask, axis=0), axis=0)
        mask_width = np.sum(np.any(mask, axis=1), axis=0)
        if mask_height < 2 or mask_width < 2:
            drop_index.append(i)
            continue
        ret_masks.append(mask)
    
    # drop out invalid class_ids
    class_ids = np.array([i for j, i in enumerate(class_ids) if j not in drop_index])

    if len(ret_masks) == 0:
        ret_image, ret_masks, class_ids = random_crop(image, masks, class_ids)
    else:
        ret_masks = np.transpose(np.asarray(ret_masks), (1, 2, 0))
    return ret_image, ret_masks, class_ids

def random_right_angle_rotate(image, masks):
    degree = random.choice(range(4))
    image = np.rot90(image, degree)
    masks = np.rot90(masks, degree)
    return image, masks

def random_brightness_transform(image, limit=[0.5,1.5]):
    image = image.copy()
    alpha = np.random.uniform(limit[0], limit[1])
    image = alpha*image
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image
