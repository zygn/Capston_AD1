import random
import cv2
import numpy as np
import os
from tqdm import tqdm
        



original_dir = './original'
maps = {}
for i in os.scandir(original_dir):
    if i.is_dir():
        for j in os.scandir(i):
            if j.is_file() and j.name.endswith('png'):
                maps[f'{j.name}'] = j.path

obstacle_dir = os.path.abspath('./obstacle')



for files in maps:

    file_name = maps[files]
    img = cv2.imread(file_name, -1)

    dst = img.copy()
    h, w = img.shape
    epsilon = 0.9999

    # 그레이스케일과 바이너리 스케일 변환
    th = cv2.bitwise_not(dst)
    contours, _ = cv2.findContours(th, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    contours.sort(key=len)
    size_contours = []
    contours_remap = []
    print("i found contour(s):", end="")
    for i in contours: 
        if len(i) >= 10:
            size_contours.append(len(i))
            contours_remap.append(i)
    print(size_contours)
    print("epslion: ", epsilon)

 
    contours = contours_remap
    # 외곽선 검출
    in_contour = contours[0]
    out_contour = contours[-1]

    # 바깥쪽 라인 
    cv2.drawContours(dst, [out_contour], -1, 100, 3)
    cv2.fillPoly(dst, [out_contour], 100)
    # 안쪽 라인 
    cv2.drawContours(dst, [in_contour], -1, 200, 3)
    cv2.fillPoly(dst, [in_contour], 200)


    # 트랙안에서 점찍기
    pbar = tqdm(range(w))
    for i in pbar:
        for j in range(h):
            if np.all(dst[i][j] == np.array(100)) and random.random() >= epsilon:
                cv2.circle(img, (j,i), 2, (0,0,0), -1)
                pbar.set_description(f"\u001b[36mAdded Obstacle - [{j},{i}]\u001b[0m ")
                # print(f"added obs: [{j},{i}]   \r", end="")

    cv2.imwrite(f"{obstacle_dir}/obs_{files}", img)

# cv2.imshow('cont', dst)
# cv2.imshow('obs_img', img)
# cv2.waitKey()
# cv2.destroyAllWindows()