from rest_framework import serializers

from . import models


class CreateDownloadModelSerializer(serializers.ModelSerializer):
    photo = serializers.ImageField()

    class Meta:
        model = models.DownloadModel
        fields = ['photo', ]


class DownloadModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.DownloadModel
        fields = '__all__'


class ParamTypeModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ParamTypeModel
        fields = '__all__'


class DocumentsTypeModelSerializer(serializers.ModelSerializer):
    param_types = serializers.SerializerMethodField()

    class Meta:
        model = models.DocumentsTypeModel
        fields = '__all__'

    def get_param_types(self, obj):
        selected_param_types = [
            i.param_type for i in models.ParamTypeDocumentsTypeModel.objects.filter(document_type=obj)
        ]
        return ParamTypeModelSerializer(selected_param_types, many=True).data


class ParamTypeDocumentsTypeModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ParamTypeDocumentsTypeModel
        fields = '__all__'


class ResultModelSerializer(serializers.ModelSerializer):
    param = serializers.SerializerMethodField()

    class Meta:
        model = models.ResultModel
        fields = '__all__'
        depth = 1

    def get_param(self, obj):
        selected_param = models.ParamValueModel.objects.filter(result=obj)
        return ParamValueModelSerializer(selected_param, many=True).data


class ParamValueModelSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.ParamValueModel
        fields = ['param_type', 'value']
        depth = 2
