from skimage import io
import numpy as np
from PIL import Image
from skimage import transform
import matplotlib.pyplot as plt
import sys
file_path = sys.argv[1]

# # ----------------- draw avg pic
# avg = np.zeros(1080000)
# for i in range(415):
#     print('reading ',i ,'/414')
#     file_name = file_path + str(i) + '.jpg'
#     img = io.imread(file_name)
#     res = img.flatten()
#     print(res.dtype)
#     avg = avg + res/415
#     # ori = np.append(ori,res)
#     # # print(ori.shape)
#     # if avg_face is None:
#     #     avg_face = np.array(res)
#     # else:
#     #     avg_face += res

#     # avg_data = np.append(avg_data,res)
#     # result = res - np.mean(res)
#     # data = np.append(data,result)
#     # avg_mean = np.array(avg_data).mean(axis=0, keepdims=True)
#     # avg_ori = np.array(avg_data)
#     # avg_data = avg_ori - avg_mean
#     # # img = Image.fromarray(avg_mean.astype(np.uint8) , 'L')

# avg = np.array(np.reshape(avg,[600,600,3]),dtype=np.uint8)
# io.imsave('avg_face.jpg',avg)
# # ----------------

data = []
weight = []
targetPic = sys.argv[2]
resizeSize = 600
generatedImg = np.zeros(resizeSize*resizeSize*3)
k = 4
for i in range(415):
    print('reading ',i ,'/414')
    file_name = file_path + '/' + str(i) + '.jpg'
    img = io.imread(file_name)
    new_img = img
    res = new_img.flatten()
    data = np.append(data,res)
print(data.shape)
X = np.reshape(data,[-1,resizeSize*resizeSize*3])
X_mean = np.mean(X)
U, s, V = np.linalg.svd((X - X_mean).T, full_matrices=False)
print(U.shape)

file_name = file_path + '/' + targetPic
img = io.imread(file_name)
new_img = img
target = new_img.flatten()
target = target-X_mean

for idx in range(k):
    ori = np.array(np.reshape(U[:,idx],[resizeSize,resizeSize,3]))
    eigenface = ori
    eigenface -= np.min(ori)
    eigenface /= np.max(eigenface)
    eigenface = (eigenface*255).astype(np.uint8)
    # io.imsave('eigenface'+str(idx)+'.jpg',eigenface)
    eigenface_minus = -ori
    eigenface_minus -= np.min(eigenface_minus)
    eigenface_minus /= np.max(eigenface_minus)
    eigenface_minus = (eigenface_minus*255).astype(np.uint8)
    # io.imsave('eigenface*-1_'+str(idx)+'.jpg',eigenface_minus)
    weight = np.append(weight,np.dot(target,U[:,idx]))

for idx in range(k):
    generatedImg += U[:,idx]*weight[idx]
    # print(np.sum(s))
    # 5383.71637743
    # print(s[idx])
    # 396.512383386
    # 199.906933537
    # 151.76995039
    # 123.458163746
print('X_mean = ',X_mean)
generatedImg += X_mean
generatedImg -= np.min(generatedImg)
generatedImg /= np.max(generatedImg)
generatedImg = (generatedImg*255).astype(np.uint8)
io.imsave('reconstruction.jpg',np.reshape(generatedImg,[resizeSize,resizeSize,3]))
