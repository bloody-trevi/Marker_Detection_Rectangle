import platform
import ctypes
import numpy as np


class myCamCapture():
    """
        oCam 1CGN-U-T 카메라를 열어 영상을 불러오는 클래스.
    """

    CTRL_BRIGHTNESS = ctypes.c_int(1)
    CTRL_CONTRAST = ctypes.c_int(2)
    CTRL_HUE = ctypes.c_int(3)
    CTRL_SATURATION = ctypes.c_int(4)
    CTRL_EXPOSURE = ctypes.c_int(5)
    CTRL_GAIN = ctypes.c_int(6)
    CTRL_WB_BLUE = ctypes.c_int(7)
    CTRL_WB_RED = ctypes.c_int(8)

    def __init__(self):
        try:
            if platform.architecture()[0] == '64bit':
                self.mydll = ctypes.cdll.LoadLibrary("./libCamCap-amd64.dll")
                # self.mydll = ctypes.CDLL(".\\libCamCap-amd64.dll")
            else:
                self.mydll = ctypes.cdll.LoadLibrary(".\\libCamCap-x86.dll")
            self.mydll.CamGetDeviceInfo.restype = ctypes.c_char_p
            self.mydll.CamOpen.restype = ctypes.POINTER(ctypes.c_int)
        except WindowsError as Error:
            print(Error)
            raise Exception('libCamCap-amd64.dll or libCamCap-x86.dll not found')

        self.cam = None
        self.resolution = (0, 0)

    def GetConnectedCamNumber(self):
        """
        연결된 캠의 번호를 반환한다.

        :return: 연결된 캠의 번호를 int로 반환
        """
        return int(self.mydll.GetConnectedCamNumber())

    def CamGetDeviceInfo(self, devno):
        """
        디바이스 번호에 해당하는 디바이스의 정보를 출력한다.

        :param devno: 디바이스 번호
        :return: USB 포트 위치, 시리얼 번호, 제품 이름, 펌웨어 버전을 반환
        """
        info = dict()
        for idx, param in enumerate(('USB_Port', 'SerialNo', 'oCamName', 'FWVersion')):
            info[param] = self.mydll.CamGetDeviceInfo(int(devno), idx + 1)
        return info

    def CamGetDeviceList(self):
        """
        디바이스 번호를 이용해 얻은 정보들을 리스트에 저장해 반환한다.

        :return: 디바이스들의 리스트를 반환
        """
        CamCount = self.GetConnectedCamNumber()
        DeviceList = list()
        for idx in range(CamCount):
            dev = self.CamGetDeviceInfo(idx)
            dev['devno'] = idx
            DeviceList.append(dev)
        return DeviceList

    def CamStart(self):
        """
        카메라가 영상을 받을 수 있게 준비한다.

        :return: 열려 있는 카메라가 없으면 None
        """
        if self.cam == None: return
        ret = self.mydll.CamStart(self.cam)

    def CamGetImage(self):
        """
        열려 있는 카메라에서 이미지를 받는다.

        :return: 열려 있는 카메라가 없으면 None. 이미지를 받았다면 (True, 받은 이미지), 읽지 못했다면 (False, None)
        """
        if self.cam == None: return 0, None
        ret = self.mydll.CamGetImage(self.cam, ctypes.c_char_p(self.bayer.ctypes.data))
        if ret == 1:
            return True, self.bayer
        else:
            return False, None

    def CamStop(self):
        """
        카메라를 정지 시킨다.

        :return: 열려 있는 카메라가 없으면 None
        """
        if self.cam == None: return
        ret = self.mydll.CamStop(self.cam)

    def CamClose(self):
        """
        열었던 카메라를 닫는다. 닫은 카메라는 None으로 초기화한다.

        :return: 열려 있는 카메라가 없으면 None
        """
        if self.cam == None: return
        ret = self.mydll.CamClose(ctypes.byref(self.cam))
        self.cam = None

    def CamGetCtrl(self, ctrl):
        """
        카메라 설정을 읽어 온다.

        :param ctrl: 컨트롤
        :return:
        """
        if self.cam == None: return
        val = ctypes.c_int()
        ret = self.mydll.CamGetCtrl(self.cam, ctrl, ctypes.byref(val))
        return val.value

    def CamSetCtrl(self, ctrl, value):
        if self.cam == None: return
        val = ctypes.c_int()
        val.value = value
        ret = self.mydll.CamSetCtrl(self.cam, ctrl, val)

    def CamGetCtrlRange(self, ctrl):
        if self.cam == None: return
        val_min = ctypes.c_int()
        val_max = ctypes.c_int()
        ret = self.mydll.CamGetCtrlRange(self.cam, ctrl, ctypes.byref(val_min), ctypes.byref(val_max))
        return val_min.value, val_max.value

    def CamOpen(self, **options):
        DevNo = options.get('DevNo')
        FramePerSec = options.get('FramePerSec')
        Resolution = options.get('Resolution')
        BytePerPixel = options.get('BytePerPixel')

        try:
            devno = DevNo
            (h, w) = Resolution
            pixelsize = BytePerPixel
            fps = FramePerSec
            self.resolution = (w, h)
            self.cam = self.mydll.CamOpen(ctypes.c_int(devno), ctypes.c_int(w), ctypes.c_int(h), ctypes.c_double(fps),
                                          0, 0)
            self.bayer = np.zeros((h, w, pixelsize), dtype=np.uint8)
            return True
        except WindowsError:
            return False


CTRL_PARAM = {
    'Brightness': myCamCapture.CTRL_BRIGHTNESS,
    'Contrast': myCamCapture.CTRL_CONTRAST,
    'Hue': myCamCapture.CTRL_HUE,
    'Saturation': myCamCapture.CTRL_SATURATION,
    'Exposure': myCamCapture.CTRL_EXPOSURE,
    'Gain': myCamCapture.CTRL_GAIN,
    'WB Blue': myCamCapture.CTRL_WB_BLUE,
    'WB Red': myCamCapture.CTRL_WB_RED
}


def adjust_gamma(image, gamma=1.0):
    """
    이미지의 감마를 조정한다. build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values

    :param image: 조정할 이미지
    :param gamma: 조정할 감마값
    :return: 조정한 이미지
    """
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)


# ROI 크기 기본값
r_ROI = 20


def check_midP(gray, start_x, start_y):
    """

    1. Harris corner detection을 이용해 이미지에서 꼭짓점을 찾는다.
    2. 발견한 꼭짓점 중에서 사각형으로 인정된 4개의 점만 저장한다.

    사각형 인식 과정:
        1. 임의의 점을 첫 번째 꼭짓점으로 저장하고 거리가 가장 먼 꼭짓점을 두 번째 꼭짓점으로 저장한다.
        2. 이 두 꼭짓점과 이루는 넓이가 가장 큰 꼭짓점을 세 번째 꼭짓점으로 저장한다.
        3. 저장한 세 꼭짓점과 이루는 넓이가 가장 큰 꼭짓점을 네 번째 꼭짓점으로 저장한다.
        4. 저장한 네 개의 꼭짓점들을 반환한다.

    :param gray: 그레이스케일 이미지
    :param start_x: ROI의 시작점의 x값
    :param start_y: ROI의 시작점의 y값
    :return: 사각형의 네 꼭짓점의 좌표
    """
    # cornerHarris parameters
    blockSize = 4
    ksize = 3
    k = 0.04

    # Harris corner detection
    gray = np.float32(gray)
    try:
        dst = cv2.cornerHarris(gray, blockSize, ksize, k)  ##
    except:
        return np.zeros((4, 2))
    dst = cv2.dilate(dst, None)
    __, dst = cv2.threshold(dst, 0.05 * dst.max(), 255, 0)
    dst = np.uint8(dst)

    # find centroids
    __, __, __, centroids = cv2.connectedComponentsWithStats(dst)

    # define the criteria to stop and refine the corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(gray, np.float32(centroids), (5, 5), (-1, -1), criteria)

    # 

    #   res에는 발견한 점들이 [y,x,y,x] 행이 여러 개 있다
    #   앞의 두 x,y는 내부 코너,
    #   뒤의 두 x,y는 외부 코너임
    #   Z 모양 순서로 저장되어 있음
    res = np.hstack((centroids, corners))
    res = np.int0(res)

    # ----------------------------------------------

    # Make lists, x and y
    x = []
    y = []
    for r in res:
        x.append(r[2])
        y.append(r[3])

    # Start with a random point
    curr_index = 0
    # The number of points and last 4 points
    n = len(x)
    pt = [0, 0, 0, 0]

    # curr_index 번째 점에서 가장 먼 점을 첫 번째 꼭지점으로 설정
    # 첫 번째 꼭지점에서부터 가장 멀리 떨어진 점을 두 번째 꼭지점으로 설정
    for i in range(0, 2):
        maxdist = 0
        for j in range(n):
            if j == curr_index: continue

            dist = abs(x[j] - x[curr_index]) + abs(y[j] - y[curr_index])
            if dist > maxdist:
                maxdist = dist
                pt[i] = j
        curr_index = pt[i]

    # 첫 번째와 두 번째 꼭지점과 나머지 점 중 하나를 선택하여,
    # 이들 세 점으로 삼각형을 만들어 넓이가 가장 크게 되는 점을 세 번째 꼭지점으로 설정한다.
    x1 = x[pt[0]]
    y1 = y[pt[0]]
    x3 = x[pt[1]]
    y3 = y[pt[1]]
    x3y1 = x3 * y1
    x1y3 = x1 * y3

    maxarea = 0
    for i in range(n):
        if i == pt[0] or i == pt[1]:
            continue

        x2 = x[i]
        y2 = y[i]
        area = abs(x1 * y2 + x2 * y3 + x3y1 - x2 * y1 - x3 * y2 - x1y3)
        if area > maxarea:
            maxarea = area
            pt[2] = i

    # 세 꼭지점들과 나머지 점 중 하나의 점을 선택하여,
    # 이들 네 점으로 사각형을 만들어 넓이가 가장 크게 되는 점을 네 번째 꼭지점으로 설정한다.
    x2 = x[pt[2]]
    y2 = y[pt[2]]
    x1y2 = x1 * y2
    x2y1 = x2 * y1
    x3y1 = x3 * y1
    x1y3 = x1 * y3
    x2y3 = x2 * y3
    x3y2 = x3 * y2
    maxarea = 0
    for i in range(n):
        if i == pt[0] or i == pt[1] or i == pt[2]:
            continue
        x4 = x[i]
        y4 = y[i]
        area = abs((x1y2 + x2 * y4 + x4 * y1) - (x2y1 + x4 * y2 + x1 * y4))
        + abs((x1 * y4 + x4 * y3 + x3y1) - (x4 * y1 + x3 * y4 + x1y3))
        + abs((x4 * y2 + x2y3 + x3 * y4) - (x2 * y4 + x3y2 + x4 * y3))
        if area > maxarea:
            maxarea = area
            pt[3] = i

    markerPt = np.zeros((4, 2))
    for i in range(len(pt)):
        markerPt[i][0] = x[pt[i]] + start_x - r_ROI
        markerPt[i][1] = y[pt[i]] + start_y - r_ROI

    return markerPt


minArea = 100.0
maxArea = 100000.0


def find_contours(img, color):
    """
    주어진 이미지에서 조건에 맞는 컨투어를 찾아낸다.

    :param img: 컨투어를 찾을 이미지
    :param color: 찾은 컨투어를 표시할 BGR 이미지
    :return: 찾은 컨투어중 볼록 사각형이고 정해진 범위내의 크기를 가지는 컨투어들
    """
    #
    img_blur = cv2.GaussianBlur(img, (5, 5), 0)
    __, binary = cv2.threshold(img_blur, 125, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #

    # Morphology
    kernel = np.ones((4, 4), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(color, contours, -1, (0, 255, 0), 1)
    try:
        # Compare the hierarchy
        hier = []
        for i in range(len(hierarchy[0])):
            for j in range(1, len(hierarchy[0])):
                if hierarchy[0][j][3] == i and hierarchy[0][j][2] == -1:
                    hier.append(contours[i])

        # Convex?
        filtered1 = []
        for cnt in hier:

            # approximation
            epsilon = 0.05 * cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, epsilon, True)

            # four points, convex
            if (cv2.isContourConvex(approx) and
                    len(approx) == 4 and
                    minArea < abs(cv2.contourArea(approx)) < maxArea):
                filtered1.append(approx)
    except:
        return []

    return filtered1


def setROI(img, contours):
    """
    ROI(Region of Interest)를 설정해준다.

    :param img: 원본 이미지
    :param contours: ROI를 선택할 기준 좌표들
    :return: 잘라낸 이미지와 그 이미지의 왼쪽 위의 좌표
    """
    start_y = contours[0][0][1]
    start_x = contours[0][0][0]
    y = start_y
    x = start_x
    for i in range(1, len(contours)):
        nextY = contours[i][0][1]
        nextX = contours[i][0][0]

        if nextY > y:  # max y
            y = nextY
        elif start_y > nextY:  # min y
            start_y = nextY

        if nextX > x:  # max x
            x = nextX
        elif start_x > nextX:  # min x
            start_x = nextX

    img_cut = img[start_y - r_ROI:y + r_ROI, start_x - r_ROI:x + r_ROI]
    return img_cut, start_x, start_y


count = False


def camCalib():
    """
    캘리브레이션 함수(사용 안함)

    :return: 내부 파라미터중 camera matrix, distortion coefficient vector
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6 * 7, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    # images = glob.glob('*.png')
    img = cv2.imread('7x6.png')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        # img = cv2.drawChessboardCorners(img, (7,6), corners2,ret)
        # cv2.imshow('img',img)
        # cv2.waitKey(0)
        global count
        count = True

    ret, cameraMtx, distC, __, __ = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return cameraMtx, distC


def cam_params():
    """
    카메라 내부 파라미터를 캘리브레이션 프로그램을 이용해 구한 값으로 설정한다.
    마커는 한 변이 187.0mm 인 정사각형으로 가정한다.
    또한 마커의 꼭짓점 저장 순서는
    1 2
    3 4
    로 저장된다.

    참고: https://darkpgmr.tistory.com/32

    :return: 마커 꼭짓점들의 실제 3D 좌표 벡터, camera matrix, distortion coefficient vector
    """
    cameraMtx = np.array([[941.156, 0, 615.588], [0, 942.199, 540.078], [0, 0, 1]])
    distCoff = np.array([-0.428858, 0.203900, -0.003420, -0.003488])

    # 마커 사이즈. 단위는 [mm]
    marker_width = 187.0  # 174.0
    marker_height = 187.0
    '''
    1 2
    4 3
    '''
    object_points = np.array([[-marker_width / 2, -marker_height / 2, 0], [marker_width / 2, -marker_height / 2, 0],
                              [marker_width / 2, marker_height / 2, 0], [-marker_width / 2, marker_height / 2, 0]])
    return object_points, cameraMtx, distCoff


def get_angle(markerPt):
    """
    check_midP 함수로 정해진 마커의 꼭짓점들을 꼭짓점들의 중점과 각 점들의 각도를 이용해 점들을 정렬한다.

    :param markerPt: check_midP로 정해진 마커의 꼭짓점들의 좌표 벡터
    :return: 정렬된 꼭짓점들의 좌표 벡터, 꼭짓점들의 중점 좌표, 중점과 각 꼭짓점들의 각도 벡터
    """
    cx = (markerPt[0][0] + markerPt[1][0] + markerPt[2][0] + markerPt[3][0]) / 4
    cy = (markerPt[0][1] + markerPt[1][1] + markerPt[2][1] + markerPt[3][1]) / 4
    cp = np.array([cx, cy])

    angle = []
    for pt in markerPt:
        angle.append((math.atan2(pt[1] - cy, pt[0] - cx), pt))

    angle = sorted(angle, key=lambda x: x[0])

    markerPt2 = np.array([angle[0][1], angle[1][1], angle[2][1], angle[3][1]])

    return markerPt2, cp, angle


def main(object_points, cameraMtx, distCoff):
    """
    이 시스템의 main loop, 다음 과정을 중단될때까지 반복한다.

    loop:
        1. myCamCapture 클래스를 이용해 oCam에서 프레임을 읽어온다.
        2. 읽어온 프레임의 밝기를 낮춘다.
        3. 프레임에서 firstContour 함수를 이용해 조건에 맞는 사각형을 찾는다.
            3-1. 찾은 사각형이 2개 이상이면 다음 프레임을 읽어온다.
        4. 찾은 사각형의 대각선의 교점이 일치하면 마커로 인정한다.
        5. 마커의 꼭짓점들을 시계 방향으로 정렬한 후, 중점과 함께 꼭짓점들을 표시한다.
        6. solvePnP 함수를 이용해 마커의 실제 좌표를 구한다.

    :param object_points: 마커의 실제 크기를 반영한 3D 좌표계에서의 꼭짓점 좌표 벡터
    :param cameraMtx: 카메라 내부 파라미터, camera matrix
    :param distCoff: distortion coefficient
    :return: 없음
    """
    cap = myCamCapture()
    if cap.GetConnectedCamNumber() == 0:
        print("oCam not Found...")
    else:
        print(cap.CamGetDeviceInfo(0))
        # print (cap.CamGetDeviceList())
        cap.CamOpen(DevNo=0, Resolution=(960, 1280), FramePerSec=30.0, BytePerPixel=1)

        start_time = time.time()
        cap.CamStart()

        cap.CamSetCtrl(myCamCapture.CTRL_EXPOSURE, -11)
        cap.CamSetCtrl(myCamCapture.CTRL_GAIN, 255)
        count = 0

        data = np.array([[0, 0, 0]])

        while (True):
            try:
                ret, frame = cap.CamGetImage()  # frame: Gray
            except:
                print('error')
            if ret is False:
                continue

            count += 1
            # -----------------------------------#
            color = cv2.cvtColor(frame, cv2.COLOR_BAYER_GB2BGR)  # BGR
            # 이미지 밝기 낮추기, b의 크기에 따라 변화(최대 255)
            b = 50
            M = np.ones(color.shape, dtype="uint8") * b
            color = cv2.subtract(color, M)

            gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)  # gray

            # 사각형 찾기
            firstContours = find_contours(gray, color)

            # 중점 판단
            if len(firstContours) <= 1:  # 마커가 여러 개가 아니기 때문에 마커가 하나일 때만 중점 찾음
                for cnt in firstContours:

                    # ROI 설정
                    img_cut, start_x, start_y = setROI(gray, cnt)

                    # 중점 비교
                    # try:
                    markerPt = check_midP(img_cut, start_x, start_y)
                    # except:
                    #    break
                    if markerPt is np.zeros((4, 2)):
                        continue

                    # 중점과 네 꼭짓점의 각도를 이용해 점을 정렬
                    markerPt2, cp, angle = get_angle(markerPt)
                    color = cv2.circle(color, (int(cp[0]), int(cp[1])), 10, (255, 0, 255), -1)

                    c = 0
                    for pt in markerPt2:
                        c = c + 1
                        text = str(int(angle[c - 1][0] * 57.295))
                        color = cv2.putText(color, text, (int(pt[0]) + 20, int(pt[1])), cv2.FONT_HERSHEY_PLAIN, 1,
                                            (255, 0, 0), 2)
                        if c == 1:
                            color = cv2.circle(color, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)  # 마커 꼭지점 표시
                        elif c == 2:
                            color = cv2.circle(color, (int(pt[0]), int(pt[1])), 10, (0, 255, 0), -1)  # 마커 꼭지점 표시
                        elif c == 3:
                            color = cv2.circle(color, (int(pt[0]), int(pt[1])), 10, (255, 0, 0), -1)  # 마커 꼭지점 표시
                        elif c == 4:
                            color = cv2.circle(color, (int(pt[0]), int(pt[1])), 10, (0, 255, 255), -1)  # 마커 꼭지점 표시

                    # Find the location of the marker
                    # try:
                    retP, rvec, tvec = cv2.solvePnP(object_points, markerPt2, cameraMtx, distCoff)
                    tvec_save = np.array([[tvec[0][0], tvec[1][0], tvec[2][0]]])
                    data = np.append(data, tvec_save, axis=0)
                    mat, _ = cv2.Rodrigues(rvec)
                    # mat = np.matrix(mat)
                    # tv = -mat.I * tvec

                    # except:
                    #    break
                    # 위치 표시
                    text = 'Location: [' + str(int(tvec[0][0])) + ', ' + str(int(tvec[1][0])) + ', ' + str(
                        int(tvec[2][0])) + ']'
                    color = cv2.putText(color, text, (20, 20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

            cv2.imshow('final test', color)

            key = cv2.waitKey(1)
            if key == 27:  # Quit with ESC
                break

        # saving the locations
        np.save('data', data)
        end_time = time.time()

        print('FPS= ', count / (end_time - start_time))
        cv2.destroyAllWindows()
        cap.CamStop()
        cap.CamClose()


# ----------------------------------------------------------------------
if __name__ == '__main__':
    import cv2
    import time
    import math

    '''
    # 카메라 파라미터들 global로 저장, 최초 동작시에만 계산 #

    global cameraMtx, distCoff
    if count == False:
        cameraMtx, distCoff = camCalib()


    # objpoints = np.array([[0, marker_height, 0], [marker_width, marker_height, 0], [marker_width, 0, 0], [0,0,0]])
    '''

    # 카메라 내부 파라미터 저장
    object_points, cameraMtx, distCoff = cam_params()

    # 메인 루프
    main(object_points, cameraMtx, distCoff)
