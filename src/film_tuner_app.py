#!/usr/bin/env python3
from __future__ import annotations

from queue import Empty, Queue
from threading import Thread
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from film_emulator import (
    FilmSettings,
    available_presets,
    emulate_film,
    read_image,
    save_settings,
    load_settings,
    settings_for_preset,
    write_image,
)


PREVIEW_MAX_DIMENSION = 1200
PREVIEW_DEBOUNCE_MS = 140


class FilmTunerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Film Tuner")
        self.geometry("1320x840")
        self.minsize(980, 640)

        self.original_image: np.ndarray | None = None
        self.preview_source: np.ndarray | None = None
        self.rendered_preview: Image.Image | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None
        self.current_path: Path | None = None

        self.render_queue: Queue = Queue()
        self.result_queue: Queue = Queue()
        self.render_generation = 0
        self.debounce_job: str | None = None
        self.loading_controls = False

        self.preset_var = tk.StringVar(value="portra")
        self.lut_path_var = tk.StringVar(value="")
        self.seed_var = tk.StringVar(value="")
        self.show_advanced_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Open a photo to begin.")

        defaults = settings_for_preset("portra")
        self.slider_vars: dict[str, tk.DoubleVar] = {
            "saturation": tk.DoubleVar(value=defaults.saturation),
            "contrast": tk.DoubleVar(value=defaults.contrast),
            "fade": tk.DoubleVar(value=defaults.fade),
            "bloom": tk.DoubleVar(value=defaults.bloom),
            "halation": tk.DoubleVar(value=defaults.halation),
            "vignette": tk.DoubleVar(value=defaults.vignette),
            "grain": tk.DoubleVar(value=defaults.grain),
            "bloom_threshold": tk.DoubleVar(value=defaults.bloom_threshold),
            "bloom_blur_sigma": tk.DoubleVar(value=defaults.bloom_blur_sigma),
            "halation_threshold": tk.DoubleVar(value=defaults.halation_threshold),
            "halation_blur_sigma": tk.DoubleVar(value=defaults.halation_blur_sigma),
            "grain_size": tk.DoubleVar(value=defaults.grain_size),
            "grain_color": tk.DoubleVar(value=defaults.grain_color),
        }
        self.matrix_vars: list[list[tk.DoubleVar]] = [
            [tk.DoubleVar(value=defaults.color_matrix[row][col]) for col in range(3)]
            for row in range(3)
        ]

        self._build_ui()
        self._wire_traces()
        self._set_controls_from_settings(defaults, schedule=False)

        self.worker = Thread(target=self._render_worker, daemon=True)
        self.worker.start()
        self.after(50, self._poll_results)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)

        preview_frame = ttk.Frame(self, padding=(10, 10, 6, 10))
        preview_frame.grid(row=0, column=0, sticky="nsew")
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.preview_canvas = tk.Canvas(
            preview_frame,
            bg="#171717",
            highlightthickness=0,
            relief="flat",
        )
        self.preview_canvas.grid(row=0, column=0, sticky="nsew")
        self.preview_canvas.bind("<Configure>", lambda _event: self._draw_preview())

        status = ttk.Label(preview_frame, textvariable=self.status_var, anchor="w")
        status.grid(row=1, column=0, sticky="ew", pady=(8, 0))

        side = ttk.Frame(self, padding=(6, 10, 10, 10), width=380)
        side.grid(row=0, column=1, sticky="ns")
        side.grid_propagate(False)
        side.columnconfigure(0, weight=1)
        side.rowconfigure(1, weight=1)

        actions = ttk.Frame(side)
        actions.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for column in range(2):
            actions.columnconfigure(column, weight=1)

        ttk.Button(actions, text="Open Photo", command=self._open_photo).grid(
            row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6)
        )
        ttk.Button(actions, text="Save Image", command=self._save_processed_image).grid(
            row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6)
        )
        ttk.Button(actions, text="Load Preset", command=self._load_preset).grid(
            row=1, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(actions, text="Save Preset", command=self._save_preset).grid(
            row=1, column=1, sticky="ew", padx=(4, 0)
        )

        control_canvas = tk.Canvas(side, highlightthickness=0)
        scrollbar = ttk.Scrollbar(side, orient="vertical", command=control_canvas.yview)
        self.controls_frame = ttk.Frame(control_canvas)
        self.controls_frame.bind(
            "<Configure>",
            lambda _event: control_canvas.configure(scrollregion=control_canvas.bbox("all")),
        )
        control_window = control_canvas.create_window((0, 0), window=self.controls_frame, anchor="nw")
        control_canvas.configure(yscrollcommand=scrollbar.set)
        control_canvas.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        control_canvas.bind(
            "<Configure>",
            lambda event: control_canvas.itemconfigure(control_window, width=event.width),
        )

        self._build_core_controls(self.controls_frame)
        self._build_advanced_controls(self.controls_frame)

    def _build_core_controls(self, parent: ttk.Frame) -> None:
        core = ttk.Labelframe(parent, text="Core", padding=10)
        core.pack(fill="x", pady=(0, 10))
        core.columnconfigure(1, weight=1)

        ttk.Label(core, text="Preset").grid(row=0, column=0, sticky="w", pady=(0, 8))
        preset_combo = ttk.Combobox(
            core,
            textvariable=self.preset_var,
            values=available_presets(),
            state="readonly",
        )
        preset_combo.grid(row=0, column=1, sticky="ew", pady=(0, 8))
        preset_combo.bind("<<ComboboxSelected>>", self._on_preset_selected)

        ttk.Button(core, text="Reset to Preset", command=self._reset_to_preset).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(0, 10)
        )

        rows = [
            ("saturation", "Saturation", 0.0, 2.0),
            ("contrast", "Contrast", 0.5, 1.8),
            ("fade", "Fade", 0.0, 0.25),
            ("bloom", "Bloom", 0.0, 0.8),
            ("halation", "Halation", 0.0, 0.8),
            ("vignette", "Vignette", 0.0, 0.7),
            ("grain", "Grain", 0.0, 0.2),
        ]
        for index, (key, label, minimum, maximum) in enumerate(rows, start=2):
            self._add_slider(core, index, key, label, minimum, maximum)

        toggle = ttk.Checkbutton(
            core,
            text="Show advanced controls",
            variable=self.show_advanced_var,
            command=self._toggle_advanced,
        )
        toggle.grid(row=len(rows) + 2, column=0, columnspan=2, sticky="w", pady=(10, 0))

    def _build_advanced_controls(self, parent: ttk.Frame) -> None:
        self.advanced_frame = ttk.Labelframe(parent, text="Advanced", padding=10)
        self.advanced_frame.columnconfigure(1, weight=1)

        row = 0
        matrix_frame = ttk.Frame(self.advanced_frame)
        matrix_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=(0, 12))
        for col, label in enumerate(("R", "G", "B"), start=1):
            ttk.Label(matrix_frame, text=label, anchor="center").grid(row=0, column=col, sticky="ew")
        for row_index, label in enumerate(("R out", "G out", "B out"), start=1):
            ttk.Label(matrix_frame, text=label).grid(row=row_index, column=0, sticky="w", padx=(0, 6))
            for col_index in range(3):
                spin = ttk.Spinbox(
                    matrix_frame,
                    textvariable=self.matrix_vars[row_index - 1][col_index],
                    from_=-0.5,
                    to=1.5,
                    increment=0.01,
                    width=7,
                    format="%.3f",
                )
                spin.grid(row=row_index, column=col_index + 1, sticky="ew", padx=2, pady=2)

        row += 1
        advanced_rows = [
            ("bloom_threshold", "Bloom Threshold", 0.0, 1.0),
            ("bloom_blur_sigma", "Bloom Blur", 0.0, 40.0),
            ("halation_threshold", "Halation Threshold", 0.0, 1.0),
            ("halation_blur_sigma", "Halation Blur", 0.0, 40.0),
            ("grain_size", "Grain Size", 0.5, 4.0),
            ("grain_color", "Grain Color", 0.0, 1.0),
        ]
        for key, label, minimum, maximum in advanced_rows:
            self._add_slider(self.advanced_frame, row, key, label, minimum, maximum)
            row += 1

        ttk.Label(self.advanced_frame, text="Seed").grid(row=row, column=0, sticky="w", pady=(8, 2))
        ttk.Entry(self.advanced_frame, textvariable=self.seed_var).grid(
            row=row, column=1, sticky="ew", pady=(8, 2)
        )
        row += 1

        ttk.Label(self.advanced_frame, text="LUT").grid(row=row, column=0, sticky="w", pady=(8, 2))
        lut_frame = ttk.Frame(self.advanced_frame)
        lut_frame.grid(row=row, column=1, sticky="ew", pady=(8, 2))
        lut_frame.columnconfigure(0, weight=1)
        ttk.Entry(lut_frame, textvariable=self.lut_path_var).grid(row=0, column=0, sticky="ew")
        ttk.Button(lut_frame, text="Browse", command=self._browse_lut).grid(row=0, column=1, padx=(6, 0))
        ttk.Button(lut_frame, text="Clear", command=lambda: self.lut_path_var.set("")).grid(
            row=0, column=2, padx=(6, 0)
        )

    def _add_slider(
        self,
        parent: ttk.Frame,
        row: int,
        key: str,
        label: str,
        minimum: float,
        maximum: float,
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=1, sticky="ew", pady=4)
        frame.columnconfigure(0, weight=1)

        scale = ttk.Scale(
            frame,
            variable=self.slider_vars[key],
            from_=minimum,
            to=maximum,
            orient="horizontal",
        )
        scale.grid(row=0, column=0, sticky="ew", padx=(0, 8))
        entry = ttk.Entry(frame, textvariable=self.slider_vars[key], width=7)
        entry.grid(row=0, column=1)

    def _wire_traces(self) -> None:
        for var in self.slider_vars.values():
            var.trace_add("write", self._on_control_changed)
        for row in self.matrix_vars:
            for var in row:
                var.trace_add("write", self._on_control_changed)
        self.seed_var.trace_add("write", self._on_control_changed)
        self.lut_path_var.trace_add("write", self._on_control_changed)

    def _on_preset_selected(self, _event: tk.Event | None = None) -> None:
        current = self._collect_settings(allow_invalid_seed=True)
        preset_defaults = settings_for_preset(self.preset_var.get())
        preset_defaults.bloom = current.bloom
        preset_defaults.bloom_threshold = current.bloom_threshold
        preset_defaults.bloom_blur_sigma = current.bloom_blur_sigma
        preset_defaults.halation = current.halation
        preset_defaults.halation_threshold = current.halation_threshold
        preset_defaults.halation_blur_sigma = current.halation_blur_sigma
        preset_defaults.vignette = current.vignette
        preset_defaults.grain = current.grain
        preset_defaults.grain_size = current.grain_size
        preset_defaults.grain_color = current.grain_color
        preset_defaults.seed = current.seed
        preset_defaults.lut_path = current.lut_path
        self._set_controls_from_settings(preset_defaults)

    def _reset_to_preset(self) -> None:
        self._set_controls_from_settings(settings_for_preset(self.preset_var.get()))

    def _toggle_advanced(self) -> None:
        if self.show_advanced_var.get():
            self.advanced_frame.pack(fill="x", pady=(0, 10))
        else:
            self.advanced_frame.pack_forget()

    def _on_control_changed(self, *_args: object) -> None:
        if self.loading_controls:
            return
        self._schedule_render()

    def _set_controls_from_settings(self, settings: FilmSettings, schedule: bool = True) -> None:
        self.loading_controls = True
        self.preset_var.set(settings.preset)
        self.lut_path_var.set(settings.lut_path or "")
        self.seed_var.set("" if settings.seed is None else str(settings.seed))

        for key in self.slider_vars:
            self.slider_vars[key].set(float(getattr(settings, key)))

        for row in range(3):
            for col in range(3):
                self.matrix_vars[row][col].set(float(settings.color_matrix[row][col]))

        self.loading_controls = False
        if schedule:
            self._schedule_render()

    def _collect_settings(self, allow_invalid_seed: bool = False) -> FilmSettings:
        seed_text = self.seed_var.get().strip()
        seed: int | None
        if not seed_text:
            seed = None
        else:
            try:
                seed = int(seed_text)
            except ValueError:
                if allow_invalid_seed:
                    seed = None
                else:
                    raise ValueError("Seed must be a whole number")

        return FilmSettings(
            preset=self.preset_var.get(),
            lut_path=self.lut_path_var.get().strip() or None,
            saturation=float(self.slider_vars["saturation"].get()),
            contrast=float(self.slider_vars["contrast"].get()),
            fade=float(self.slider_vars["fade"].get()),
            color_matrix=[
                [float(self.matrix_vars[row][col].get()) for col in range(3)]
                for row in range(3)
            ],
            bloom=float(self.slider_vars["bloom"].get()),
            bloom_threshold=float(self.slider_vars["bloom_threshold"].get()),
            bloom_blur_sigma=float(self.slider_vars["bloom_blur_sigma"].get()),
            halation=float(self.slider_vars["halation"].get()),
            halation_threshold=float(self.slider_vars["halation_threshold"].get()),
            halation_blur_sigma=float(self.slider_vars["halation_blur_sigma"].get()),
            vignette=float(self.slider_vars["vignette"].get()),
            grain=float(self.slider_vars["grain"].get()),
            grain_size=float(self.slider_vars["grain_size"].get()),
            grain_color=float(self.slider_vars["grain_color"].get()),
            seed=seed,
        )

    def _open_photo(self) -> None:
        path = filedialog.askopenfilename(
            title="Open photo",
            filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        try:
            self.original_image = read_image(path)
        except Exception as exc:
            messagebox.showerror("Open Photo", str(exc))
            return

        self.current_path = Path(path)
        self.preview_source = self._downscale_for_preview(self.original_image)
        h, w = self.original_image.shape[:2]
        ph, pw = self.preview_source.shape[:2]
        self.status_var.set(f"Opened {self.current_path.name} ({w}x{h}); preview {pw}x{ph}.")
        self._schedule_render(immediate=True)

    def _browse_lut(self) -> None:
        path = filedialog.askopenfilename(
            title="Open .cube LUT",
            filetypes=[("Cube LUT", "*.cube"), ("All files", "*.*")],
        )
        if path:
            self.lut_path_var.set(path)

    def _save_preset(self) -> None:
        try:
            settings = self._collect_settings()
        except Exception as exc:
            messagebox.showerror("Save Preset", str(exc))
            return

        initial = "film-preset.json"
        if self.current_path:
            initial = f"{self.current_path.stem}-film-preset.json"
        path = filedialog.asksaveasfilename(
            title="Save preset",
            defaultextension=".json",
            initialfile=initial,
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            save_settings(path, settings)
        except Exception as exc:
            messagebox.showerror("Save Preset", str(exc))
            return
        self.status_var.set(f"Saved preset: {Path(path).name}")

    def _load_preset(self) -> None:
        path = filedialog.askopenfilename(
            title="Load preset",
            filetypes=[("JSON preset", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return

        try:
            settings = load_settings(path)
        except Exception as exc:
            messagebox.showerror("Load Preset", str(exc))
            return

        self._set_controls_from_settings(settings)
        self.status_var.set(f"Loaded preset: {Path(path).name}")

    def _save_processed_image(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Save Image", "Open a photo first.")
            return

        try:
            settings = self._collect_settings()
        except Exception as exc:
            messagebox.showerror("Save Image", str(exc))
            return

        if self.current_path:
            initial = f"{self.current_path.stem}_film{self.current_path.suffix}"
        else:
            initial = "film-output.jpg"

        path = filedialog.asksaveasfilename(
            title="Save processed image",
            defaultextension=Path(initial).suffix or ".jpg",
            initialfile=initial,
            filetypes=[
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
                ("TIFF", "*.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not path:
            return

        source = self.original_image
        self.status_var.set("Rendering full-resolution image...")
        Thread(
            target=self._export_worker,
            args=(source, settings, path),
            daemon=True,
        ).start()

    def _export_worker(self, source: np.ndarray, settings: FilmSettings, path: str) -> None:
        try:
            out = emulate_film(source, settings=settings)
            write_image(path, out)
        except Exception as exc:
            self.result_queue.put(("export_error", str(exc)))
            return
        self.result_queue.put(("export_done", path))

    def _downscale_for_preview(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        largest = max(h, w)
        if largest <= PREVIEW_MAX_DIMENSION:
            return img.copy()

        scale = PREVIEW_MAX_DIMENSION / largest
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _schedule_render(self, immediate: bool = False) -> None:
        if self.preview_source is None:
            return

        if self.debounce_job is not None:
            self.after_cancel(self.debounce_job)
            self.debounce_job = None

        if immediate:
            self._queue_render()
        else:
            self.debounce_job = self.after(PREVIEW_DEBOUNCE_MS, self._queue_render)

    def _queue_render(self) -> None:
        self.debounce_job = None
        if self.preview_source is None:
            return

        try:
            settings = self._collect_settings()
        except Exception as exc:
            self.status_var.set(str(exc))
            return

        self.render_generation += 1
        generation = self.render_generation
        self.status_var.set("Rendering preview...")
        self.render_queue.put((generation, self.preview_source, settings))

    def _render_worker(self) -> None:
        while True:
            job = self.render_queue.get()
            if job is None:
                return

            while True:
                try:
                    newer_job = self.render_queue.get_nowait()
                except Empty:
                    break
                if newer_job is None:
                    return
                job = newer_job

            generation, source, settings = job
            try:
                out = emulate_film(source, settings=settings)
            except Exception as exc:
                self.result_queue.put(("preview_error", generation, str(exc)))
                continue
            self.result_queue.put(("preview_done", generation, out))

    def _poll_results(self) -> None:
        while True:
            try:
                result = self.result_queue.get_nowait()
            except Empty:
                break

            kind = result[0]
            if kind == "preview_done":
                _kind, generation, image = result
                if generation == self.render_generation:
                    self._show_preview(image)
                    self.status_var.set("Preview updated.")
            elif kind == "preview_error":
                _kind, generation, error = result
                if generation == self.render_generation:
                    self.status_var.set(error)
            elif kind == "export_done":
                _kind, path = result
                self.status_var.set(f"Saved image: {Path(path).name}")
                messagebox.showinfo("Save Image", f"Saved image:\n{path}")
            elif kind == "export_error":
                _kind, error = result
                messagebox.showerror("Save Image", error)
                self.status_var.set("Image export failed.")

        self.after(50, self._poll_results)

    def _show_preview(self, img: np.ndarray) -> None:
        array = (np.clip(img, 0.0, 1.0) * 255).astype(np.uint8)
        self.rendered_preview = Image.fromarray(array)
        self._draw_preview()

    def _draw_preview(self) -> None:
        self.preview_canvas.delete("all")
        width = max(1, self.preview_canvas.winfo_width())
        height = max(1, self.preview_canvas.winfo_height())

        if self.rendered_preview is None:
            self.preview_canvas.create_text(
                width // 2,
                height // 2,
                text="Open a photo",
                fill="#d7d7d7",
                font=("TkDefaultFont", 18),
            )
            return

        image_w, image_h = self.rendered_preview.size
        scale = min(width / image_w, height / image_h)
        display_w = max(1, int(image_w * scale))
        display_h = max(1, int(image_h * scale))
        resized = self.rendered_preview.resize((display_w, display_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(resized)
        self.preview_canvas.create_image(
            width // 2,
            height // 2,
            image=self.preview_photo,
            anchor="center",
        )

    def _on_close(self) -> None:
        self.render_queue.put(None)
        self.destroy()


def main() -> None:
    app = FilmTunerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
