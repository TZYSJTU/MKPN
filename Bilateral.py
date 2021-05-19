import cv2

for i in range(449):

    img=cv2.imread('D:/2021-2/dkn-master/dkn-master/Compare/Bicubic/{}_bic.jpg'.format(i))

    #cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)
    #9 邻域直径，两个 75 分别是空间高斯函数标准差，灰度值相似性高斯函数标准差
    blur1 = cv2.bilateralFilter(img,15,100,30)
    # blur1 = cv2.GaussianBlur(img,(11,11),10)
    cv2.imwrite('D:/2021-2/dkn-master/dkn-master/Compare/JBU/NYU/{}.jpg'.format(i), blur1)
    # cv2.imwrite('G.jpg', blur2)
    # cv2.imshow('G',blur2)
    cv2.imshow('Bilateral',blur1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
