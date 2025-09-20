"""Tkinter based desktop application for the realtime translation pipeline."""
from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk
from typing import Dict, Optional

import sounddevice as sd

try:  # Support running both as `python -m client.gui_app` and `python gui_app.py`
    from .translator import Translator, TranslatorEvent  # type: ignore[attr-defined]
except ImportError:  # Fallback when executed as a standalone script
    from translator import Translator, TranslatorEvent


class TranslatorGUI:
    """User interface wrapper for the :class:`Translator` pipeline."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Danish → Russian Speech Translator")
        self.root.geometry("760x640")

        self.event_queue: "queue.Queue[TranslatorEvent]" = queue.Queue()
        self.translator: Optional[Translator] = None
        self.active_model: Optional[str] = None

        self.is_running = False

        self.microphone_var = tk.StringVar()
        self.model_var = tk.StringVar(value="base")
        self.partial_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Готов к запуску")
        self.phrase_count_var = tk.StringVar(value="0")
        self.latest_latency_var = tk.StringVar(value="—")
        self.average_latency_var = tk.StringVar(value="—")
        self.audible_var = tk.StringVar(value="Не слышно")

        self.volume_level = tk.DoubleVar(value=0.0)

        self._build_ui()
        self._populate_microphones()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(100, self._process_events)

    # ------------------------------------------------------------------
    # UI creation helpers
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        mic_label = ttk.Label(top_frame, text="Микрофон:")
        mic_label.grid(row=0, column=0, sticky=tk.W, padx=(0, 8))

        self.microphone_combo = ttk.Combobox(
            top_frame, textvariable=self.microphone_var, state="readonly", width=50
        )
        self.microphone_combo.grid(row=0, column=1, sticky=tk.W)

        refresh_button = ttk.Button(
            top_frame, text="Обновить", command=self._populate_microphones
        )
        refresh_button.grid(row=0, column=2, padx=(8, 0))

        model_label = ttk.Label(top_frame, text="Модель Whisper:")
        model_label.grid(row=1, column=0, sticky=tk.W, pady=(8, 0))

        model_combo = ttk.Combobox(
            top_frame,
            textvariable=self.model_var,
            state="readonly",
            values=["base", "small", "medium"],
        )
        model_combo.grid(row=1, column=1, sticky=tk.W, pady=(8, 0))

        button_frame = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        button_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(
            button_frame, text="Start", command=self.start_translation
        )
        self.start_button.grid(row=0, column=0, padx=(0, 10))

        self.stop_button = ttk.Button(
            button_frame, text="Stop", command=self.stop_translation, state=tk.DISABLED
        )
        self.stop_button.grid(row=0, column=1)

        status_frame = ttk.LabelFrame(self.root, text="Статус", padding=10)
        status_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Label(status_frame, textvariable=self.status_var).grid(
            row=0, column=0, columnspan=2, sticky=tk.W
        )

        ttk.Label(status_frame, text="Громкость микрофона:").grid(
            row=1, column=0, sticky=tk.W, pady=(8, 0)
        )
        self.volume_bar = ttk.Progressbar(
            status_frame, orient=tk.HORIZONTAL, maximum=1.0, variable=self.volume_level
        )
        self.volume_bar.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=(8, 0))

        self.audible_label = ttk.Label(status_frame, textvariable=self.audible_var)
        self.audible_label.grid(row=1, column=2, padx=(12, 0), sticky=tk.W)

        ttk.Label(status_frame, text="Обработано фраз:").grid(
            row=2, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(status_frame, textvariable=self.phrase_count_var).grid(
            row=2, column=1, sticky=tk.W, pady=(8, 0)
        )

        ttk.Label(status_frame, text="Последняя задержка, мс:").grid(
            row=3, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(status_frame, textvariable=self.latest_latency_var).grid(
            row=3, column=1, sticky=tk.W, pady=(8, 0)
        )

        ttk.Label(status_frame, text="Средняя задержка, мс:").grid(
            row=4, column=0, sticky=tk.W, pady=(8, 0)
        )
        ttk.Label(status_frame, textvariable=self.average_latency_var).grid(
            row=4, column=1, sticky=tk.W, pady=(8, 0)
        )

        transcription_frame = ttk.Frame(self.root, padding=10)
        transcription_frame.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.LabelFrame(
            transcription_frame, text="Распознанный датский текст", padding=8
        )
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        self.danish_text = scrolledtext.ScrolledText(
            left_frame, wrap=tk.WORD, height=12, state=tk.DISABLED
        )
        self.danish_text.pack(fill=tk.BOTH, expand=True)

        ttk.Label(left_frame, textvariable=self.partial_var, foreground="#444").pack(
            anchor=tk.W, pady=(6, 0)
        )

        right_frame = ttk.LabelFrame(
            transcription_frame, text="Русский перевод", padding=8
        )
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0))

        self.russian_text = scrolledtext.ScrolledText(
            right_frame, wrap=tk.WORD, height=12, state=tk.DISABLED
        )
        self.russian_text.pack(fill=tk.BOTH, expand=True)

    def _populate_microphones(self) -> None:
        devices = sd.query_devices()
        microphones = []
        default_input_index = sd.default.device[0]
        for index, device in enumerate(devices):
            if device.get("max_input_channels", 0) > 0:
                label = f"{index}: {device['name']}"
                microphones.append((label, index))

        if not microphones:
            self.microphone_combo["values"] = []
            self.microphone_var.set("")
            messagebox.showerror(
                "Нет устройств",
                "Не найдено ни одного входного аудиоустройства. Подключите микрофон и нажмите 'Обновить'.",
            )
            return

        labels = [item[0] for item in microphones]
        self.microphone_mapping: Dict[str, int] = {label: idx for label, idx in microphones}
        self.microphone_combo["values"] = labels

        default_label = next(
            (label for label, idx in microphones if idx == default_input_index),
            microphones[0][0],
        )
        self.microphone_var.set(default_label)

    # ------------------------------------------------------------------
    # Control handlers
    # ------------------------------------------------------------------
    def start_translation(self) -> None:
        if self.is_running:
            return

        selected_label = self.microphone_var.get()
        if not selected_label:
            messagebox.showwarning("Микрофон", "Выберите входное устройство.")
            return

        input_index = self.microphone_mapping[selected_label]
        model_name = self.model_var.get()

        self.status_var.set("Загрузка моделей...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.DISABLED)

        if self.translator and self.active_model == model_name:
            self.translator.input_device_index = input_index
            thread = threading.Thread(
                target=self._start_existing_translator, daemon=True
            )
            thread.start()
        else:
            thread = threading.Thread(
                target=self._create_and_start_translator,
                args=(input_index, model_name),
                daemon=True,
            )
            thread.start()

    def _start_existing_translator(self) -> None:
        if not self.translator:
            return
        try:
            self.translator.start()
        except Exception as exc:  # pragma: no cover - defensive
            self.event_queue.put(
                TranslatorEvent("error", {"message": str(exc)})
            )
            self.event_queue.put(
                TranslatorEvent("status", {"state": "stopped"})
            )

    def _create_and_start_translator(self, input_index: int, model_name: str) -> None:
        try:
            translator = Translator(
                event_queue=self.event_queue,
                input_device_index=input_index,
                whisper_model=model_name,
            )
            self.translator = translator
            self.active_model = model_name
            translator.start()
        except Exception as exc:  # pragma: no cover - defensive
            self.event_queue.put(
                TranslatorEvent("error", {"message": str(exc)})
            )
            self.event_queue.put(
                TranslatorEvent("status", {"state": "stopped"})
            )

    def stop_translation(self) -> None:
        if not self.translator:
            return

        self.stop_button.config(state=tk.DISABLED)

        thread = threading.Thread(target=self._stop_translator, daemon=True)
        thread.start()

    def _stop_translator(self) -> None:
        try:
            if self.translator:
                self.translator.stop()
        except Exception as exc:  # pragma: no cover - defensive
            self.event_queue.put(
                TranslatorEvent("error", {"message": str(exc)})
            )

    # ------------------------------------------------------------------
    # Event handling
    # ------------------------------------------------------------------
    def _process_events(self) -> None:
        try:
            while True:
                event = self.event_queue.get_nowait()
                self._handle_event(event)
        except queue.Empty:
            pass
        finally:
            self.root.after(100, self._process_events)

    def _handle_event(self, event: TranslatorEvent) -> None:
        if event.type == "status":
            self._handle_status(event.payload)
        elif event.type == "warning":
            self.status_var.set(event.payload.get("message", "Предупреждение"))
        elif event.type == "error":
            message = event.payload.get("message", "Произошла ошибка")
            self.status_var.set(message)
            messagebox.showerror("Ошибка", message)
            self._set_running(False)
        elif event.type == "volume":
            self._update_volume(event.payload)
        elif event.type == "partial_transcription":
            self.partial_var.set(f"Идёт распознавание: {event.payload['text']}")
        elif event.type == "final_transcription":
            self._append_text(self.danish_text, event.payload["text"])
            self.partial_var.set("")
            self.phrase_count_var.set(str(event.payload.get("count", 0)))
        elif event.type == "translation":
            self._append_text(self.russian_text, event.payload["text"])
        elif event.type == "latency":
            latest = event.payload.get("latest")
            average = event.payload.get("average")
            if latest is not None:
                self.latest_latency_var.set(f"{latest * 1000:.0f}")
            if average is not None:
                self.average_latency_var.set(f"{average * 1000:.0f}")

    def _handle_status(self, payload: Dict) -> None:
        state = payload.get("state")
        if state == "starting":
            self.status_var.set("Запуск...")
            self._set_running(True, buttons=False)
        elif state == "running":
            self.status_var.set("Работает")
            self._set_running(True)
        elif state == "stopped":
            self.status_var.set("Остановлено")
            self._set_running(False)
            self.partial_var.set("")
            self.latest_latency_var.set("—")
            self.average_latency_var.set("—")
            self.volume_level.set(0.0)
            self.audible_var.set("Не слышно")

    def _set_running(self, running: bool, buttons: bool = True) -> None:
        self.is_running = running
        if buttons:
            self.start_button.config(state=tk.DISABLED if running else tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL if running else tk.DISABLED)

    def _update_volume(self, payload: Dict) -> None:
        volume = min(max(payload.get("value", 0.0), 0.0), 1.0)
        audible = payload.get("audible", False)
        self.volume_level.set(volume)
        if audible:
            self.audible_var.set("Слышно")
            self.audible_label.configure(foreground="green")
        else:
            self.audible_var.set("Не слышно")
            self.audible_label.configure(foreground="red")

    def _append_text(self, widget: scrolledtext.ScrolledText, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, text + "\n")
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def on_close(self) -> None:
        if self.translator:
            try:
                self.translator.stop()
            except Exception:
                pass
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    TranslatorGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

