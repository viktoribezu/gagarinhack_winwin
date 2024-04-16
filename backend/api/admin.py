from django.contrib import admin

from . import models

admin.site.register(models.DownloadModel)
admin.site.register(models.ParamTypeModel)
admin.site.register(models.DocumentsTypeModel)
admin.site.register(models.ParamTypeDocumentsTypeModel)
admin.site.register(models.ResultModel)
admin.site.register(models.ParamValueModel)
