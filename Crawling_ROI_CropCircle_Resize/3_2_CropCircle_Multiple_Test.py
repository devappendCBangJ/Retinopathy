import cv2
import os
import time
import numpy as np

# 이미지 저장할 폴더 만들기
def CreateDir(dir_name):
    try:
        if not os.path.exists("./crop_data"):
            os.mkdir("./crop_data")

        # 키워드 폴더 만들기
        if not os.path.exists("./crop_data/{}".format(dir_name)):
            os.mkdir("./crop_data/{}".format(dir_name))
    except OSError:
        print("Error : 파일 생성 실패")

def Crop_Img(target):
    # 폴더 생성
    CreateDir(target)
    print(target)

    # 파일 불러오기
    file_list = os.listdir("./user_data/{}".format(target))
    print(file_list)
    # print("file_list : ", file_list) # 확인용 코드

    # 각 사진별 알고리즘 수행
    for file_name in file_list:
        # 원본 이미지 읽기 + 복사 + gray scale 변환

        read_path = "./user_data/{}/{}".format(target, file_name)

        """
        # 한글 파일만 읽기 가능
        src = cv2.imread(file_path)
        """

        # 한글 파일 or 영어 파일 전부 읽기 가능
        img_array = np.fromfile(read_path, np.uint8)
        decoded_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # decoded_img가 존재하지 않는 경우 : 다음 for 루프 반복
        if decoded_img is None:
            continue
        
        # decoded_img가 존재하는 경우
        dst = decoded_img.copy()
        gray = cv2.cvtColor(decoded_img, cv2.COLOR_BGR2GRAY)

        # print(dst) # 확인용 코드
        # print(dst.shape) # 확인용 코드

        # Hough Circle Transform
        dst_h, dst_w, dst_c = dst.shape
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, max(dst_h, dst_w), param1=250, param2=10,
                                   minRadius=int(dst_h / 4), maxRadius=int(dst_h / 2))

        # 근사 원이 존재하지 않는 경우 : 다음 for 루프 반복
        if circles is None:
            print("Doesn't have circle in : {}/{}.png !!!".format(target, pure_file_name)) # 확인용 코드
            continue

        # 근사 원이 존재하는 경우
        for circle in circles[0]:
            # circle 그리기
            # print(circle) # 확인용 코드
            cir_x, cir_y, cir_r = circle
            roi_cir_r = cir_r * (1.35)  # ROI 자르기 위한 조금 더 큰 원 반지름
            # cv2.circle(dst, (int(cir_x), int(cir_y)), int(cir_r), (255, 255, 255), 5) # 확인용 코드
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
            if (int(cir_y) - int(roi_cir_r) < 0):
                roi_cir_r -= abs(int(cir_y) - int(roi_cir_r))
            if (int(cir_y) + int(roi_cir_r) > dst_h):
                roi_cir_r -= abs((int(cir_y) + int(roi_cir_r)) - dst_h)

            # x ROI 계산
            if (int(cir_x) - int(roi_cir_r) < 0):
                roi_cir_r -= abs(int(cir_x) - int(roi_cir_r))
            if (int(cir_x) + int(roi_cir_r) > dst_w):
                roi_cir_r -= abs((int(cir_x) + int(roi_cir_r)) - dst_w)

            # ROI 자르기 + ROI 화면 표시
            roi = dst[int(cir_y) - int(roi_cir_r): int(cir_y) + int(roi_cir_r),
                  int(cir_x) - int(roi_cir_r): int(cir_x) + int(roi_cir_r)]  # 원본 이미지에서 선택 영역만 ROI로 지정
            # cv2.imshow('cropped', roi)  # ROI 지정 영역을 새 창으로 표시 # 확인용 코드
            # cv2.moveWindow('cropped', 0, 0)  # 새 창을 화면 좌측 상단으로 이동 # 확인용 코드

            # ROI 영역 파일 저장
            pure_file_name = file_name[:-4]
            # print(pure_file_name) # 확인용 코드

            write_path = "./crop_data/{}/{}.png".format(target, pure_file_name)

            """
            # 한글 파일만 쓰기 가능
            cv2.imwrite(write_path, roi) # ROI 영역만 파일로 저장
            """

            # 한글 파일 or 영어 파일 전부 쓰기 가능
            extension = os.path.splitext(write_path)[1] # 이미지 확장자
            # print("extension : ", extension) # 확인용 코드
            result, encoded_img = cv2.imencode(extension, roi)

            if result:
                with open(write_path, mode='w+b') as f:
                    encoded_img.tofile(f)
            print("Image cropped: {}/{}.png".format(target, pure_file_name)) # 확인용 코드

            # 원본 이미지 출력
            # cv2.imshow("dst", dst) # 확인용 코드

            # 종료
            # cv2.waitkey(0) # 확인용 코드
            cv2.destroyAllWindows()

            # time.sleep(3) # 확인용 코드

# 원하는 데이터 크롤링
# targets = ["태양", "real sun", "수성", "Mercury", "금성", "Venus", "지구", "earth", "화성", "mars", "목성", "Jupiter", "토성", "saturn", "천왕성", "Uranus", "해왕성", "Neptune", "명왕성", "Pluto", "농구공", "basketball"]
targets = ["태양", "수성", "금성", "지구", "화성", "목성", "토성", "천왕성", "해왕성", "명왕성", "농구공"]
for target in targets:
    Crop_Img(target)