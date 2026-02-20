# Удаление голоса из видео с помощью нейросети

Скрипт для автоматического удаления голоса из видеофайлов с использованием нейросети Demucs (htdemucs).

## Описание

Этот проект использует нейросеть Demucs для разделения аудио на отдельные источники (drums, bass, other, vocals) и автоматически удаляет вокал, оставляя только инструментальную часть.

## Примеры

### Оригинальное видео

<video width="320" height="240" controls>
  <source src="https://github.com/missisgalu/ads_no_voice/raw/main/optimized_9x16.mp4" type="video/mp4">
  Ваш браузер не поддерживает видео тег. [Скачать видео](https://github.com/missisgalu/ads_no_voice/raw/main/optimized_9x16.mp4)
</video>

[Скачать optimized_9x16.mp4](https://github.com/missisgalu/ads_no_voice/raw/main/optimized_9x16.mp4)

### Видео без голоса

<video width="320" height="240" controls>
  <source src="https://github.com/missisgalu/ads_no_voice/raw/main/optimized_9x16_no_voice.mp4" type="video/mp4">
  Ваш браузер не поддерживает видео тег. [Скачать видео](https://github.com/missisgalu/ads_no_voice/raw/main/optimized_9x16_no_voice.mp4)
</video>

[Скачать optimized_9x16_no_voice.mp4](https://github.com/missisgalu/ads_no_voice/raw/main/optimized_9x16_no_voice.mp4)

> **Примечание:** GitHub не поддерживает встроенное воспроизведение видео в README. HTML теги `<video>` могут не работать на GitHub, но будут работать при просмотре README локально или на других платформах. Используйте прямые ссылки для скачивания или просмотра видео.

## Установка

```bash
pip install -r requirements.txt
```

## Использование

```bash
python remove_voice.py <путь_к_видео> [путь_к_выходному_файлу]
```

Пример:
```bash
python remove_voice.py optimized_9x16.mp4 optimized_9x16_no_voice.mp4
```

Если не указать выходной файл, он будет создан автоматически с суффиксом `_no_voice`.

## Требования

- Python 3.8+
- FFmpeg (должен быть установлен в системе)
- PyTorch
- Demucs
- soundfile

## Как это работает

1. Извлекает аудио из видео с помощью FFmpeg
2. Разделяет аудио на источники (drums, bass, other, vocals) с помощью нейросети Demucs
3. Удаляет вокал, суммируя остальные источники
4. Объединяет видео с новым аудио без голоса

## Технические детали

- Используется модель `htdemucs` от Facebook Research
- Поддержка GPU (CUDA) и CPU
- Асинхронная обработка для высокой производительности
- Автоматическая конвертация аудио в нужный формат
