import dlib
import cv2

# 导入cnn模型
cnn_face_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')  # 调用训练好的cnn进行人脸检测
failed_frames = list()

group = '02'

for i in [118]:
    # read file and convert to gray scale image
    img = cv2.imread("/Users/leon/Downloads/Biwi Kinect Head Pose Database/hpdb/{0}/frame_00{1:03d}_rgb.png".format(group, i))  # opencv 读取图片，并显示
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # 取灰度

    # run the face detector
    rects = cnn_face_detector(img, 1)  # 进行检测
    # check if any face detected
    if len(rects) == 0:
        failed_frames.append(i)
        continue
    # print("Number of faces detected: {}".format(len(rects)))  # 打印检测到的人脸数
    max_width = 0

    for d in rects:
        # rects为含各个脸的数据的列表，每个元素是一个mmod_rectangles对象。这个对象包含有2个成员变量：dlib.rectangle类，表示对象的位置；dlib.confidence，表示置信度。
        if d.rect.width() > max_width:
            face = d.rect
            max_width = d.rect.width()

    print("Pic {}: Detection : Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, face.left(), face.top(),
                                                                                                face.right(), face.bottom(),
                                                                                                rects[0].confidence))
    # print(face.width())

    # cv2.rectangle()画出矩形,参数1：图像，参数2：矩形左上角坐标，参数3：矩形右下角坐标，参数4：画线对应的rgb颜色，参数5：线的宽度
    # cv2.rectangle(img, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

    # cv2.namedWindow("img", 2)  # #图片窗口可调节大小
    # cv2.imshow("img", img)  # 显示图像
    # cv2.waitKey(0)  # 等待按键，然后退出

    # if there is a face in frame, intercept the face from frame and save to image
    face_intercepted = img[face.top():face.bottom(), face.left():face.right()]
    # output the img to  local
    # cv2.imwrite('test.png', testImg, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    cv2.imwrite('/Users/leon/Downloads/Biwi Kinect Head Pose Database/hpdb_face_intercepted/{0}/frame_00{1:03d}_rgb.png'.format(group, i),
                face_intercepted)

print('Failed frames:{}'.format(failed_frames))
