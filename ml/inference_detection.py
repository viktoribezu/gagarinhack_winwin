# идет на python >=3.7 and <=3.11
import os
import cv2
import numpy as np
import pytesseract
from sklearn.cluster import KMeans
from ultralytics import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


IMGSZ = (1120, 1280)
# путь до папки с классифицированными изображениями
path_to_imgs = "../data_gagarin/data/"
imgs_to_text = "./results/"

img_paths = [path_to_imgs + i for i in os.listdir(path_to_imgs)]
img_to_text_paths = [imgs_to_text + i for i in os.listdir(imgs_to_text)]

model_path = r"C:\Users\ivano\Desktop\winwinhack\best_model\best.pt"

custom_config = r"tessedit_char_whitelist=0123456789 --oem 0 --psm 6 --dpi 96"
# custom_config_alpha = r"tessedit_char_whitelist=0123456789ABEKMNOPCTYX --dpi 100"   # --user-words user_word.txt


def enhance_image(image):
    r, g, b = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 2

    enhanced_r = clahe.apply(r)
    enhanced_g = clahe.apply(g)
    enhanced_b = clahe.apply(b)

    enhanced_image = cv2.merge((enhanced_r, enhanced_g, enhanced_b))
    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.5, beta=0)

    return enhanced_image


# 3 канала
def rework(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = enhance_image(image_rgb)
    pixels = image_rgb.reshape((-1, 3))
    n_colors = 2
    kmeans = KMeans(n_clusters=n_colors)
    kmeans.fit(pixels)
    main_colors = kmeans.cluster_centers_.astype(int)
    most_common_color = main_colors[np.argmin(np.bincount(kmeans.labels_))]

    treshold_post_class = 70

    mask = np.any(np.abs(pixels - most_common_color) > treshold_post_class, axis=1)
    result = np.where(mask.reshape(image.shape[:2]), 255, 0).astype(np.uint8)
    result = cv2.bitwise_not(result)

    _, result = cv2.threshold(result, 100, 255, cv2.THRESH_BINARY)  # 180
    return result


def cropp_imgs_to_text(preds, config):  #
    """
    Функция для вырезания предсказанных Bounding boxes
    из исходных изображений

    input -> предсказания
    output -> папка results с обрезанными картинками
    """
    texts = []
    texts_dict = {}
    for iter in range(len(preds)):
        img = preds[iter].orig_img
        try:
            x, y, x_1, y_1 = [
                round(i) for i in list(preds[iter].boxes.xyxy[0].to("cpu").numpy())
            ]

            roi_color = img[y:y_1, x:x_1]
            roi_color = rework(roi_color)
            # print(roi_color)
            name = preds[iter].path.split("/")[-1]
            # cv2.imwrite(f"./results/{name[:-4]}.jpg", roi_color)
            text = pytesseract.image_to_string(roi_color, lang="eng", config=config)  #
            texts.append("".join(c if c.isdigit() else "" for c in text))
            texts_dict[name] = "".join(
                c if c.isdigit() else "" for c in text
            )  # or c.isalpha()
        except:
            continue

    return texts, texts_dict


# инициализируем модель и загружаем веса
model = YOLO(model_path)
# делаем предсказания
preds = model.predict(img_paths[:50], save=True, imgsz=IMGSZ)
# вырезаем и сохраняем картинки
texts, texts_dict = cropp_imgs_to_text(preds, custom_config)  #
