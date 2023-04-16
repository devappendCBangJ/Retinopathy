import cv2

# 원본 이미지 읽기 + 복사 + gray scale 변환
src = cv2.imread("hi1.jpg")
dst = src.copy()
gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

# print(dst) # 확인용 코드
# print(dst.shape) # 확인용 코드

# Hough Circle Transform
dst_h, dst_w, dst_c = dst.shape
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, max(dst_h, dst_w), param1 = 250, param2 = 10, minRadius = int(dst_h/4), maxRadius = int(dst_h/2))

for circle in circles[0]:
    # circle 그리기
    # print(circle) # 확인용 코드
    cir_x, cir_y, cir_r = circle
    roi_cir_r = cir_r * (1.35) # ROI 자르기 위한 조금 더 큰 원 반지름
    cv2.circle(dst, (int(cir_x), int(cir_y)), int(cir_r), (255, 255, 255), 5)
    # print(cir_x) # 확인용 코드

    """
    # y ROI 계산
    if(int(cir_y) - int(roi_cir_r) < 0):
        y1 = 0
        roi_cir_r -= abs(int(cir_y) - int(roi_cir_r))
    else:
        y1 = int(cir_y) - int(roi_cir_r)
    if(int(cir_y) + int(roi_cir_r) > dst_h):
        y2 = dst_h
        roi_cir_r -= abs((int(cir_y) + int(roi_cir_r)) - dst_h)
    else:
        y2 = int(cir_y) + int(roi_cir_r)

    # x ROI 계산
    if(int(cir_x) - int(roi_cir_r) < 0):
        x1 = 0
        roi_cir_r -= abs(int(cir_x) - int(roi_cir_r))
    else:
        x1 = int(cir_x) - int(roi_cir_r)
    if(int(cir_x) + int(roi_cir_r) > dst_w):
        # x2 = dst_w
        roi_cir_r -= abs((int(cir_x) + int(roi_cir_r)) - dst_w)
    else:
        x2 = int(cir_x) + int(roi_cir_r)
    """

    # y ROI 계산
    if(int(cir_y) - int(roi_cir_r) < 0):
        roi_cir_r -= abs(int(cir_y) - int(roi_cir_r))
    if(int(cir_y) + int(roi_cir_r) > dst_h):
        roi_cir_r -= abs((int(cir_y) + int(roi_cir_r)) - dst_h)

    # x ROI 계산
    if(int(cir_x) - int(roi_cir_r) < 0):
        roi_cir_r -= abs(int(cir_x) - int(roi_cir_r))
    if(int(cir_x) + int(roi_cir_r) > dst_w):
        roi_cir_r -= abs((int(cir_x) + int(roi_cir_r)) - dst_w)

    # 화면 표시 on 좌측 상단
    roi = dst[int(cir_y) - int(roi_cir_r) : int(cir_y) + int(roi_cir_r), int(cir_x) - int(roi_cir_r) : int(cir_x) + int(roi_cir_r)] # 원본 이미지에서 선택 영역만 ROI로 지정
    # roi = dst[y1:y2, x1:x2]  # 원본 이미지에서 선택 영역만 ROI로 지정
    cv2.imshow('cropped', roi) # ROI 지정 영역을 새 창으로 표시
    cv2.moveWindow('cropped', 0, 0) # 새 창을 화면 좌측 상단으로 이동

    # ROI 영역 파일 저장
    cv2.imwrite('hi1_cropped.jpg', roi) # ROI 영역만 파일로 저장

    print('cropped.') # 확인용 코드

# 원본 이미지 출력
cv2.imshow("dst", dst)

# 종료
cv2.waitKey(0)
cv2.destroyAllWindows()