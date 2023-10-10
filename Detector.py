import cv2, time, os, tensorflow as tf
import numpy as np

from tensorflow.python.keras.utils.data_utils import get_file

np.random.seed(42) 


class Detector:
    def __init__(self):
        pass
    
    def readClasses(self, classesFilePath):
        # Đọc danh sách các lớp từ file
        with open(classesFilePath, 'r') as f:
            self.classesList = f.read().splitlines()
        # Tạo danh sách màu ngẫu nhiên cho từng lớp
        self.colorList = np.random.uniform(low=0, high=255, size=(len(self.classesList), 3))

    def downloadModel(self, modelURL):
        # Tải model từ URL
        fileName = os.path.basename(modelURL)
        self.modelName = fileName[:fileName.index('.')]
        self.cacheDir = "./pretrained_models"
        os.makedirs(self.cacheDir, exist_ok=True)
        get_file(fname=fileName, origin=modelURL, cache_dir=self.cacheDir, cache_subdir="checkpoints", extract=True)

    def loadModel(self):
        # Tải model đã tải xuống
        print("Loading model..." + self.modelName)
        tf.keras.backend.clear_session()
        self.model = tf.saved_model.load(os.path.join(self.cacheDir, "checkpoints", self.modelName, "saved_model"))
        print ("Model" + self.modelName + "loaded successfully!")
    
    def createBoundingBox(self, image, threshold=0.5):
        # Chuẩn bị dữ liệu đầu vào cho model
        inputTensor = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        inputTensor = tf.convert_to_tensor(inputTensor)
        inputTensor = inputTensor[tf.newaxis, ...]

        # Dự đoán bounding box và các thông tin liên quan
        detections = self.model(inputTensor)
        bboxs = detections['detection_boxes'][0].numpy()
        classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
        classScores = detections['detection_scores'][0].numpy()

        imH, imW, imC = image.shape

        # Sử dụng non-max suppression để loại bỏ các bounding box trùng lặp
        bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50, iou_threshold=threshold, score_threshold=threshold)

        print(bboxIdx)
        # Vẽ bounding box và hiển thị nhãn lớp
        if len(bboxIdx) != 0:
            for i in bboxIdx:
                bbox = tuple(bboxs[i].tolist())
                classConfidnece = round(100*classScores[i])
                classIndex = classIndexes[i]
                classLabelText = self.classesList[classIndex]
                classColor = self.colorList[classIndex]
                displayText = '{}: {}%'.format(classLabelText, classConfidnece)
                ymin, xmin, ymax, xmax = bbox  # bounding box được trả về theo thứ tự ymin, xmin, ymax, xmax
                xmin, xmax, ymin, ymax = (xmin * imW, xmax * imW, ymin * imH, ymax * imH) # chuyển về tọa độ pixel
                xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax) # chuyển về kiểu int

                # Vẽ bounding box và hiển thị nhãn lớp
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=classColor, thickness=1) 
                cv2.putText(image, displayText, (xmin, ymin-10), cv2.FONT_HERSHEY_COMPLEX, 1, classColor, 2)
        return image

    def predictImage(self, imagePath, threshold=0.5):
        # Dự đoán trên một ảnh và resize ảnh về kích thước 1080x600
        image = cv2.imread(imagePath)
        imH, imW, imC = image.shape
        image = cv2.resize(image, (600, 850))
        bboxImage = self.createBoundingBox(image, threshold)
        cv2.imshow("Result", bboxImage)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def predictVideo(self, videoPath, threshold=0.5):
        cap = cv2.VideoCapture(videoPath)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('resized_video.avi', fourcc, 30, (original_width // 2, original_height // 2))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized_frame = cv2.resize(frame, (original_width // 2, original_height // 2))
            bboxImage = self.createBoundingBox(resized_frame, threshold)
            cv2.imshow("Result", bboxImage)
            out.write(bboxImage)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def predictWebcam(self, threshold=0.5):
        # Dự đoán trên webcam
        cap = cv2.VideoCapture(0)
        if (cap.isOpened() == False):
            print("Error opening video stream or file")
            return
        (success, image) = cap.read()

        startTime = 0

        while success:
            currentTime = time.time()
            fps = 1 / (currentTime - startTime)
            startTime = currentTime
            bboxImage = self.createBoundingBox(image, threshold)
            cv2.imshow("Result", bboxImage)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            (success, image) = cap.read()
        cv2.destroyAllWindows()