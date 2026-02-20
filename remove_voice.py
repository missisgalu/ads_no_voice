"""
Скрипт для удаления голоса из видео с использованием нейросети Demucs.
Извлекает аудио, разделяет на вокал и инструментал, заменяет аудиодорожку без голоса.
"""

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional
import tempfile
import shutil

try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    from demucs.audio import convert_audio
    import soundfile as sf
    import numpy as np
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    print("Установите зависимости: pip install -r requirements.txt")
    sys.exit(1)


class VoiceRemover:
    """Класс для удаления голоса из видео с использованием Demucs."""
    
    def __init__(self, model_name: str = "htdemucs"):
        """
        Инициализация VoiceRemover.
        
        Args:
            model_name: Название модели Demucs (htdemucs, htdemucs_ft, mdx_extra)
        """
        self.model_name = model_name
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Используется устройство: {self.device}")
    
    async def load_model(self):
        """Асинхронная загрузка модели Demucs."""
        print(f"Загрузка модели {self.model_name}...")
        # Загрузка модели в отдельном потоке, чтобы не блокировать event loop
        loop = asyncio.get_event_loop()
        self.model = await loop.run_in_executor(
            None,
            lambda: get_model(self.model_name)
        )
        self.model.to(self.device)
        print("Модель загружена успешно")
    
    def check_ffmpeg(self) -> bool:
        """Проверка наличия FFmpeg в системе."""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    async def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """
        Извлечение аудио из видео с помощью FFmpeg.
        
        Args:
            video_path: Путь к входному видео
            audio_path: Путь для сохранения аудио
            
        Returns:
            True если успешно, False иначе
        """
        print(f"Извлечение аудио из {video_path}...")
        
        if not self.check_ffmpeg():
            print("Ошибка: FFmpeg не найден. Установите FFmpeg.")
            return False
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ffmpeg",
                        "-i", video_path,
                        "-vn",  # Без видео
                        "-acodec", "pcm_s16le",  # WAV формат
                        "-ar", "44100",  # Частота дискретизации
                        "-ac", "2",  # Стерео
                        "-y",  # Перезаписать файл если существует
                        audio_path
                    ],
                    check=True,
                    capture_output=True
                )
            )
            print(f"Аудио извлечено: {audio_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при извлечении аудио: {e.stderr.decode()}")
            return False
    
    async def separate_audio(self, audio_path: str, output_dir: str) -> Optional[str]:
        """
        Разделение аудио на вокал и инструментал с помощью Demucs.
        
        Args:
            audio_path: Путь к входному аудио
            output_dir: Директория для сохранения результатов
            
        Returns:
            Путь к файлу без вокала или None при ошибке
        """
        print(f"Разделение аудио с помощью {self.model_name}...")
        
        if self.model is None:
            await self.load_model()
        
        # Загрузка аудио
        loop = asyncio.get_event_loop()
        wav, sample_rate = await loop.run_in_executor(
            None,
            lambda: sf.read(audio_path, always_2d=True)
        )
        # Конвертация в torch tensor: [channels, samples]
        wav = torch.from_numpy(wav.T).float()
        
        # Конвертация аудио в формат модели (стерео, нужная частота дискретизации)
        model_sample_rate = getattr(self.model, 'sample_rate', 44100)
        model_channels = getattr(self.model, 'audio_channels', 2)
        
        wav = convert_audio(
            wav.unsqueeze(0),
            sample_rate,
            model_sample_rate,
            model_channels
        )
        wav = wav.to(self.device)
        
        # Разделение источников
        print("Обработка нейросетью...")
        with torch.no_grad():
            sources = await loop.run_in_executor(
                None,
                lambda: apply_model(self.model, wav, device=self.device, split=True, overlap=0.25)
            )
        
        # sources shape: [batch, sources, channels, samples]
        # Источники обычно: [drums, bass, other, vocals] для htdemucs
        # Нам нужны все кроме vocals (индексы 0, 1, 2)
        sources = sources.cpu()
        
        if sources.shape[1] >= 4:
            # Есть вокал отдельно (обычно последний источник)
            # Суммируем drums, bass, other
            no_vocals = sources[:, [0, 1, 2], :, :].sum(dim=1, keepdim=False)
        else:
            # Если модель не разделяет вокал, используем все кроме последнего
            no_vocals = sources[:, :-1, :, :].sum(dim=1, keepdim=False)
        
        # no_vocals shape: [batch, channels, samples]
        # Убираем batch dimension
        no_vocals = no_vocals.squeeze(0)  # [channels, samples]
        
        # Конвертация обратно в исходную частоту дискретизации
        model_sample_rate = getattr(self.model, 'sample_rate', 44100)
        no_vocals = convert_audio(
            no_vocals.unsqueeze(0),  # [1, channels, samples]
            model_sample_rate,
            sample_rate,
            2  # стерео
        )
        no_vocals = no_vocals.squeeze(0)  # [channels, samples]
        
        # Создание директории для результатов
        os.makedirs(output_dir, exist_ok=True)
        
        # Путь для сохранения
        output_audio = os.path.join(output_dir, "no_vocals.wav")
        
        # Сохранение: конвертация в numpy и сохранение через soundfile
        # no_vocals shape: [channels, samples] -> нужно [samples, channels]
        no_vocals_np = no_vocals.cpu().numpy().T  # [samples, channels]
        await loop.run_in_executor(
            None,
            lambda: sf.write(output_audio, no_vocals_np, sample_rate)
        )
        
        print(f"Аудио без голоса сохранено: {output_audio}")
        return output_audio
    
    async def merge_audio_video(
        self,
        video_path: str,
        audio_path: str,
        output_path: str
    ) -> bool:
        """
        Объединение видео с новым аудио без голоса.
        
        Args:
            video_path: Путь к исходному видео
            audio_path: Путь к аудио без голоса
            output_path: Путь для сохранения результата
            
        Returns:
            True если успешно, False иначе
        """
        print(f"Объединение видео и аудио...")
        
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    [
                        "ffmpeg",
                        "-i", video_path,
                        "-i", audio_path,
                        "-c:v", "copy",  # Копировать видео без перекодирования
                        "-c:a", "aac",  # Кодек аудио
                        "-map", "0:v:0",  # Видео из первого файла
                        "-map", "1:a:0",  # Аудио из второго файла
                        "-shortest",  # Обрезать по самому короткому потоку
                        "-y",  # Перезаписать если существует
                        output_path
                    ],
                    check=True,
                    capture_output=True
                )
            )
            print(f"Видео сохранено: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Ошибка при объединении: {e.stderr.decode()}")
            return False
    
    async def remove_voice_from_video(
        self,
        video_path: str,
        output_path: Optional[str] = None
    ) -> bool:
        """
        Полный процесс удаления голоса из видео.
        
        Args:
            video_path: Путь к входному видео
            output_path: Путь для сохранения результата (если None, добавляется _no_voice)
            
        Returns:
            True если успешно, False иначе
        """
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Ошибка: файл {video_path} не найден")
            return False
        
        if output_path is None:
            output_path = str(video_path.parent / f"{video_path.stem}_no_voice{video_path.suffix}")
        
        # Создание временной директории
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_audio = os.path.join(temp_dir, "input_audio.wav")
            separation_dir = os.path.join(temp_dir, "separated")
            temp_no_vocals = os.path.join(temp_dir, "no_vocals.wav")
            
            # Шаг 1: Извлечение аудио
            if not await self.extract_audio(str(video_path), temp_audio):
                return False
            
            # Шаг 2: Разделение аудио
            separated_audio = await self.separate_audio(temp_audio, separation_dir)
            if separated_audio is None:
                return False
            
            # Копируем результат в temp_no_vocals для удобства
            shutil.copy(separated_audio, temp_no_vocals)
            
            # Шаг 3: Объединение видео и нового аудио
            if not await self.merge_audio_video(
                str(video_path),
                temp_no_vocals,
                output_path
            ):
                return False
        
        print(f"\nГотово! Видео без голоса сохранено: {output_path}")
        return True


async def main():
    """Главная функция."""
    if len(sys.argv) < 2:
        print("Использование: python remove_voice.py <путь_к_видео> [путь_к_выходному_файлу]")
        print("Пример: python remove_voice.py video.mp4 video_no_voice.mp4")
        sys.exit(1)
    
    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    remover = VoiceRemover(model_name="htdemucs")
    
    success = await remover.remove_voice_from_video(video_path, output_path)
    
    if not success:
        print("\nОшибка при обработке видео")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
