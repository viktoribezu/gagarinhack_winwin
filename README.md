# Проект для Гагарин хакатона
## Как запустить
`docker-compose up --build`

Создать суперпользователя

`docker-compose exec backend python manage.py createsuperuser`

Если таблиц нет

`docker-compose exec backend python manage.py makemigrations api`
