import copy
from pathlib import Path

import django_filters.rest_framework
from rest_framework import status
from rest_framework import generics
from rest_framework import permissions
from rest_framework.response import Response

import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision
from PIL import Image
import numpy as np
import os
import cv2
import pytesseract
from sklearn.cluster import KMeans
from ultralytics import YOLO

from . import serializers
from . import models


def predict(model, test_loader):
    RUN_DEVICE = "cpu"
    with torch.no_grad():
        logits = []
        filenames = []
        for images, img_names in test_loader:
            images = images.to(RUN_DEVICE)
            model.eval()
            outputs = model(images).cpu()
            logits.append(outputs)
            filenames += img_names.cpu()
    probs = nn.functional.softmax(torch.cat(logits), dim=-1).numpy()
    predicted_classes = np.argmax(probs, axis=1)
    # predicted_classes = [class_names[idx] for idx in class_indices]
    filenames = np.array([int(tens.numpy()) for tens in filenames])
    # filenames = [class_names[idx] for idx in filenames]

    return probs, predicted_classes, filenames


def get_net(pth_file_path=None, freezing=False):
    RUN_DEVICE = "cpu"
    AMOUNT_OF_CLASSES = 4
    resnet = torchvision.models.resnet152(pretrained=True)
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, AMOUNT_OF_CLASSES)
    if pth_file_path is not None:
        resnet.load_state_dict(torch.load(pth_file_path, map_location=RUN_DEVICE))

    if freezing:
        counter = 0
        for child in resnet.children():
            if counter < 18:  # заморозка первых 18 слоев
                for param in child.parameters():
                    param.requires_grad = False
                    counter += 1
            # print(iresnet_finetuned)

    resnet = resnet.to(RUN_DEVICE)
    return resnet


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
        except Warning:
            continue

    return texts, texts_dict


class BipDataset(Dataset):  # класс датасета
    def __init__(self, files, mode=None):
        super().__init__()
        # список файлов для загрузки
        # режим работы
        # self.mode = mode

        # if self.mode not in DATA_MODES:
        #     print(f"{self.mode} is not correct; correct modes: {DATA_MODES}")
        #     raise NameError

        # self._amount_of_augmentations = AMOUNT_OF_AUGMENTATIONS
        self.files = self._get_files(files)  # sorted(files)
        self.labels = self._get_labels()
        self.len_ = len(self.files)
        # self.augmentations = augmentations

    def _get_files(self, files):
        raw_files = sorted(files)
        # if self.mode != 'test':
        #     files = [file for file in raw_files for _ in range(self._amount_of_augmentations)]
        # else:
        #     files = raw_files
        files = raw_files
        return files

    def _get_labels(self):
        raw_labels = [path.parent.name for path in self.files]  # получаем имя директории. Имя директории = имя файла
        labels_numbers = [
            0 if string == "Drivers" else 1 if string == "Passports" else 2 if string == "PTS" else 3 if string == "STS" else -1
            for string in raw_labels]

        # СЕРЕЖАААА! Нужно вот это переделать. Преобразуем классы в инты. Если паспорт, то 0, если тс, то 1 и тд. Реализуй это пж
        # if self.mode != 'test':
        #     labels = [label for label in labels_numbers for _ in range(self._amount_of_augmentations)]
        # else:
        #     labels = labels_numbers
        labels = labels_numbers
        return labels

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        file_copied = copy.deepcopy(file)
        image = Image.open(file_copied).convert('RGB')
        image.load()
        return image

    def __getitem__(self, index):
        TRANSFORM = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        RESCALE_SIZE = (224, 224)
        image = self.load_sample(self.files[index])
        image = image.resize(RESCALE_SIZE)
        image = np.array(image)

        image = np.array(image / 255, dtype='float32')
        image = TRANSFORM(image)
        # if self.mode != 'test' and not(self.augmentations is None):
        #     image = self.augmentations(image)
        label = self.labels[index]
        sample = image, label

        return sample


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
#resnet = get_net(pth_file_path='./api/models/Resnet152_ep_6_from_10.pth', freezing=False)
IMGSZ = (1120, 1280)
model_path = "./api/models/best.pt"

custom_config = r"tessedit_char_whitelist=0123456789 --oem 0 --psm 6 --dpi 96"

model = YOLO(model_path)
model_class = YOLO('./api/models/best_class.pt')


class DownloadModelListView(generics.ListCreateAPIView):
    model = models.DownloadModel
    queryset = models.DownloadModel.objects.all()
    serializer_class = serializers.DownloadModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'user__id': ["in", "exact", "icontains"],
        'status': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]

    def create(self, request, *args, **kwargs):
        class_names = ["Drivers", "Passports", "PTS", "STS"]
        serializer = serializers.CreateDownloadModelSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)
        serializer.save(user=self.request.user)
        photo_path = Path('./'+str(serializer.data["photo"]))
        print(photo_path)
        #dataset = BipDataset([photo_path, ])
        #data = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
        #classes_probs, predicted_classes, filenames = predict(resnet, data)
        class_names = {
            0: 'driver_back',
            1: 'driver_front',
            2: 'pas_1',
            3: 'pas_5',
            4: 'pas_6',
            5: 'pas_8',
            6: 'pts_back',
            7: 'pts_front',
            8: 'sts_back',
            9: 'sts_front'
        }
        results = model_class([photo_path])

        result = results[0]

        preds = model.predict([photo_path], save=True, imgsz=IMGSZ)
        # вырезаем и сохраняем картинки
        texts, texts_dict = cropp_imgs_to_text(preds, custom_config)
        data_photo = texts_dict[serializer.data["photo"].split('/')[-1]]
        models.DocumentsTypeModel.objects.get_or_create(name=class_names[int(result.boxes.cls)])
        headers = self.get_success_headers(serializer.data)

        return Response({
            'type': class_names[int(result.boxes.cls)].split('_')[0],
            'confidence': float(result.boxes.conf),
            'series': data_photo[:4],
            'number': data_photo[4:],
            'slide': class_names[int(result.boxes.cls)].split('_')[1]
        }, status=status.HTTP_200_OK, headers=headers)


class ParamTypeModelListView(generics.ListAPIView):
    queryset = models.ParamTypeModel.objects.all()
    serializer_class = serializers.ParamTypeModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'name': ["in", "exact", "icontains"]
    }
    permission_classes = [permissions.IsAuthenticated]


class DocumentsTypeModelListView(generics.ListAPIView):
    queryset = models.DocumentsTypeModel.objects.all()
    serializer_class = serializers.DocumentsTypeModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'name': ["in", "exact", "icontains"]
    }
    permission_classes = [permissions.IsAuthenticated]


class ParamTypeDocumentsTypeModelListView(generics.ListAPIView):
    queryset = models.ParamTypeDocumentsTypeModel.objects.all()
    serializer_class = serializers.ParamTypeDocumentsTypeModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'param_type__id': ["in", "exact", "icontains"],
        'document_type__id': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]


class ResultModelListView(generics.ListAPIView):
    queryset = models.ResultModel.objects.all()
    serializer_class = serializers.ResultModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'download__id': ["in", "exact", "icontains"],
        'document_type__id': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]


class ParamValueModelListView(generics.ListAPIView):
    queryset = models.ParamValueModel.objects.all()
    serializer_class = serializers.ParamValueModelSerializer
    filter_backends = [django_filters.rest_framework.DjangoFilterBackend]
    filterset_fields = {
        'param_type__id': ["in", "exact", "icontains"],
        'value': ["in", "exact", "icontains"],
        'result__id': ["in", "exact", "icontains"],
    }
    permission_classes = [permissions.IsAuthenticated]
