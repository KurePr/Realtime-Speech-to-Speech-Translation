"""Translator module that wraps the existing speech pipeline for GUI usage."""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import sounddevice as sd
import torch
from transformers import MarianMTModel, MarianTokenizer

from server.models.speech_recognition import SpeechRecognitionModel
from server.models.text_to_speech import TextToSpeechModel

from client.utils.print_audio import get_volume_norm


@dataclass
class TranslatorEvent:
    """Data structure representing events sent from :class:`Translator`."""

    type: str
    payload: dict


class Translator:
    """High level wrapper around the existing realtime translation pipeline.

    The class manages microphone input, Whisper transcription, neural machine
    translation and Text-to-Speech playback. Updates are emitted through a
    thread-safe queue that the GUI can poll.
    """

    SAMPLE_RATE = 16_000
    SAMPLE_WIDTH = 2  # Bytes for int16 samples

    def __init__(
        self,
        event_queue: "queue.Queue[TranslatorEvent]",
        input_device_index: Optional[int] = None,
        output_device_index: Optional[int] = None,
        whisper_model: str = "base",
        source_lang: str = "da",
        target_lang: str = "ru",
        translation_model_name: str = "Helsinki-NLP/opus-mt-da-ru",
    ) -> None:
        self.event_queue = event_queue
        self.input_device_index = input_device_index
        # sounddevice uses a tuple of (input, output). If an explicit output
        # device is not provided we fall back to the default from sounddevice.
        if output_device_index is None:
            default_devices = sd.default.device
            self.output_device_index = default_devices[1]
        else:
            self.output_device_index = output_device_index

        self.source_lang = source_lang
        self.target_lang = target_lang

        self.data_queue: "queue.Queue[tuple[object, bytes]]" = queue.Queue()
        self.transcriber = SpeechRecognitionModel(
            data_queue=self.data_queue,
            generation_callback=self._handle_generation,
            final_callback=self._handle_final_transcription,
            model_name=whisper_model,
        )
        # Force Whisper to transcribe in the source language instead of
        # translating to English directly so that we can display the original
        # transcript to the user.
        self.transcriber.decoding_options = {
            "task": "transcribe",
            "language": self.source_lang,
        }

        self.text_to_speech = TextToSpeechModel(
            callback_function=self._handle_synthesised_audio
        )
        self.text_to_speech.load_speaker_embeddings()

        self.translation_tokenizer = MarianTokenizer.from_pretrained(
            translation_model_name
        )
        self.translation_model = MarianMTModel.from_pretrained(
            translation_model_name
        )

        self._stop_event = threading.Event()
        self._audio_stream: Optional[sd.InputStream] = None
        self._playback_thread: Optional[threading.Thread] = None
        self._client_token = object()
        self._playback_queue: "queue.Queue[Optional[Tuple[np.ndarray, Optional[float]]]]" = queue.Queue()
        self._latency_times: "queue.Queue[float]" = queue.Queue()

        self._running = False
        self._latest_volume = 0.0
        self._last_volume_event = 0.0
        self._volume_threshold = 0.01

        self._phrase_count = 0
        self._total_latency = 0.0

        # sounddevice callbacks operate in separate threads. We cache a token so
        # that SpeechRecognitionModel treats all audio chunks as coming from the
        # same "client".
        self._client_identifier = object()

    def start(self) -> None:
        """Start processing audio and translation."""

        if self._running:
            return

        self._running = True
        self._stop_event.clear()

        self.event_queue.put(
            TranslatorEvent("status", {"state": "starting"})
        )

        # Reset statistics
        self._phrase_count = 0
        self._total_latency = 0.0

        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:  # pragma: no cover - defensive
                break

        self.transcriber.start(self.SAMPLE_RATE, self.SAMPLE_WIDTH)

        self._audio_stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="float32",
            device=self.input_device_index,
            callback=self._audio_callback,
        )
        self._audio_stream.start()

        self._playback_thread = threading.Thread(
            target=self._playback_worker, daemon=True
        )
        self._playback_thread.start()

        self.event_queue.put(
            TranslatorEvent("status", {"state": "running"})
        )

    def stop(self) -> None:
        """Stop all background processing."""

        if not self._running:
            return

        self._running = False
        self._stop_event.set()

        if self._audio_stream is not None:
            self._audio_stream.stop()
            self._audio_stream.close()
            self._audio_stream = None

        sd.stop()

        self.transcriber.stop()

        # Drain latency information
        while not self._latency_times.empty():
            try:
                self._latency_times.get_nowait()
            except queue.Empty:  # pragma: no cover - defensive
                break

        # Drain playback queue and notify worker to exit
        self._playback_queue.put(None)
        if self._playback_thread is not None:
            self._playback_thread.join()
            self._playback_thread = None

        self.event_queue.put(
            TranslatorEvent("status", {"state": "stopped"})
        )

    # ------------------------------------------------------------------
    # Internal helper methods
    # ------------------------------------------------------------------
    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if not self._running:
            return

        if status:
            self.event_queue.put(
                TranslatorEvent("warning", {"message": str(status)})
            )

        volume = float(get_volume_norm(indata))
        previous_volume = self._latest_volume
        self._latest_volume = volume
        current_time = time.time()
        if (
            current_time - self._last_volume_event > 0.2
            or abs(volume - previous_volume) > 0.05
        ):
            self._last_volume_event = current_time
            self.event_queue.put(
                TranslatorEvent(
                    "volume",
                    {
                        "value": volume,
                        "audible": volume > self._volume_threshold,
                    },
                )
            )

        audio_int16 = np.clip(indata, -1.0, 1.0)
        audio_int16 = (audio_int16 * np.iinfo(np.int16).max).astype(np.int16)
        self.data_queue.put((self._client_identifier, audio_int16.tobytes()))

    def _handle_generation(self, packet: dict) -> None:
        if not self._running:
            return

        text = packet.get("text", "").strip()
        if not text:
            return

        self.event_queue.put(
            TranslatorEvent("partial_transcription", {"text": text})
        )

    def _handle_final_transcription(self, text: str, _client_socket) -> None:
        if not self._running:
            return

        text = (text or "").strip()
        if not text:
            return

        self._phrase_count += 1
        phrase_time = time.time()
        self._latency_times.put(phrase_time)

        self.event_queue.put(
            TranslatorEvent(
                "final_transcription",
                {"text": text, "count": self._phrase_count},
            )
        )

        try:
            translation = self._translate_text(text)
        except Exception as exc:  # pragma: no cover - defensive
            self._discard_pending_latency()
            self.event_queue.put(
                TranslatorEvent(
                    "error",
                    {
                        "message": f"Translation error: {exc}",
                    },
                )
            )
            return

        if translation:
            self.event_queue.put(
                TranslatorEvent(
                    "translation", {"text": translation, "language": self.target_lang}
                )
            )
            self.text_to_speech.synthesise(translation, self._client_token)
        else:
            self._discard_pending_latency()

    def _translate_text(self, text: str) -> str:
        inputs = self.translation_tokenizer(
            text, return_tensors="pt", padding=True
        )
        with torch.no_grad():
            generated_tokens = self.translation_model.generate(**inputs)
        return self.translation_tokenizer.decode(
            generated_tokens[0], skip_special_tokens=True
        )

    def _handle_synthesised_audio(self, audio_tensor: torch.Tensor, _client_socket) -> None:
        if not self._running:
            return

        audio = audio_tensor.detach().cpu().numpy().astype(np.float32)
        phrase_time = None
        try:
            phrase_time = self._latency_times.get_nowait()
        except queue.Empty:
            pass

        self._playback_queue.put((audio, phrase_time))

    def _playback_worker(self) -> None:
        while True:
            try:
                item = self._playback_queue.get(timeout=0.1)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if item is None:
                self._playback_queue.task_done()
                break

            audio, phrase_time = item
            if not self._running:
                self._playback_queue.task_done()
                continue
            try:
                playback_start = time.time()
                sd.play(
                    audio,
                    samplerate=self.SAMPLE_RATE,
                    device=self.output_device_index,
                    blocking=True,
                )
                sd.wait()
                if phrase_time is not None and self._phrase_count > 0:
                    latency = playback_start - phrase_time
                    self._total_latency += latency
                    average_latency = self._total_latency / max(
                        self._phrase_count, 1
                    )
                    self.event_queue.put(
                        TranslatorEvent(
                            "latency",
                            {
                                "latest": latency,
                                "average": average_latency,
                            },
                        )
                    )
            except Exception as exc:  # pragma: no cover - defensive
                self.event_queue.put(
                    TranslatorEvent(
                        "error", {"message": f"Playback error: {exc}"}
                    )
                )
            finally:
                self._playback_queue.task_done()

    def _discard_pending_latency(self) -> None:
        try:
            self._latency_times.get_nowait()
        except queue.Empty:
            pass

