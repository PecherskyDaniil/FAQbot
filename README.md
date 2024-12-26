# Описание
Данный бот отвечает на вопросы об ИГУ. В репозитории леджат две версии реализации этого бота: vtensorflow - это бот с multilanguagebert на tensorflow (обратите внимание что для него может потребоваться очень конкретная версия tensorflow, python и сопутствующих библиотек), а vtorch - это загруженный с помощью torch RUBERT. <ins>САМ БОТ НАПИСАН НА TENSORFLOW</ins>.

Обязательно при использовании git clone пропишите опцию --recurse-submodules, иначе DeepPavlov/rubert-base-cased не загрузиться. Скачивание репозитория может занять больше времени чем обычно (минут 5).
```
git clone --recurse-submodules https://github.com/PecherskyDaniil/FAQbot
```

Перед запуском файла  chatbot.py или chatbottorch.py можно обучить моедли на парах вопрос-ответ в файле "QA.json". Для запуска обучения запустите файл aitrain.py или aitraintorch.py (в запсиимости от того какого бота вы хотиет обучить заново) и дождитесь выполения программы. В результате скрипт перезапишет фалы dataset и model в своей папке.

**<ins>Я бы настоятельно рекомендовал использовать версию с rubert на pytorch в папке vtorch, потому что она работает без интернета и использует именно RUBERT</ins>**.
