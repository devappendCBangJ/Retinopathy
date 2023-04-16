import cv2

src = cv2.imread("Jupiter1.jpg", cv2.IMREAD_COLOR)

dst = cv2.resize(src, dsize=(224, 224), interpolation=cv2.INTER_AREA)
dst2 = cv2.resize(src, dsize=(0, 0), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)

cv2.imwrite("Jupyter1_resize.png", dst) # ROI 영역만 파일로 저장

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.imshow("dst2", dst2)
cv2.waitKey()
cv2.destroyAllWindows()