from django.urls import path, include

from . import hello_world_view
from . import login_view
from . import views

urlpatterns = [
    path('hello_world/', hello_world_view.HelloView.as_view()),
    path('login/', login_view.LoginView.as_view()),
    path('downloads/', views.DownloadModelListView.as_view()),
    path('param_types/', views.ParamTypeModelListView.as_view()),
    path('documents_types/', views.DocumentsTypeModelListView.as_view()),
    path('param_types_documents_types/', views.ParamTypeDocumentsTypeModelListView.as_view()),
    path('results/', views.ResultModelListView.as_view()),
    path('param_values/', views.ParamValueModelListView.as_view()),
]
