#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Independent GIMP 3 Semantic Segmentation development plugin."""

import sys
import tempfile
import traceback
from pathlib import Path

import gi

gi.require_version("Gimp", "3.0")
gi.require_version("GimpUi", "3.0")
gi.require_version("Gtk", "3.0")

from gi.repository import Gimp, GimpUi, GObject, Gio, GLib, Gtk

PLUGIN_ID = "semsegnew"
MENU_LABEL = "Semantic Segmentation (Development)"

MODEL_DPT = "DPT-Hybrid (Default)"
MODEL_SEGFORMER = "SegFormer-B0"

# Resolve the project from this source file, not the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_CONFIG_FILE = PROJECT_ROOT / "local-config.json"

import json

if not LOCAL_CONFIG_FILE.is_file():
    raise FileNotFoundError(
        f"Local configuration not found: {LOCAL_CONFIG_FILE}"
    )

with LOCAL_CONFIG_FILE.open("r", encoding="utf-8") as config_file:
    LOCAL_CONFIG = json.load(config_file)

PYTHON_PATHS = LOCAL_CONFIG["python"]

# Dedicated Python environments: do not mix model dependencies.
DPT_ADAPTER_FILE = PROJECT_ROOT / "models" / "dpt_hybrid" / "adapter.py"
SEGFORMER_ADAPTER_FILE = PROJECT_ROOT / "models" / "segformer_b0" / "adapter.py"

DPT_PYTHON_EXE = Path(PYTHON_PATHS["dpt_cpu"])
DPT_GPU_PYTHON_EXE = Path(PYTHON_PATHS["dpt_gpu"])
SEGFORMER_PYTHON_EXE = Path(PYTHON_PATHS["segformer_cpu"])
SEGFORMER_GPU_PYTHON_EXE = Path(PYTHON_PATHS["segformer_gpu"])


class SemanticSegmentationPlugin(Gimp.PlugIn):

    def do_query_procedures(self):
        return [PLUGIN_ID]

    def do_set_i18n(self, procedure_name):
        return False, "gimp30-python", None

    def do_create_procedure(self, name):
        if name != PLUGIN_ID:
            return None

        procedure = Gimp.ImageProcedure.new(
            self,
            name,
            Gimp.PDBProcType.PLUGIN,
            self.run,
            None,
        )
        procedure.set_image_types("*")
        procedure.set_sensitivity_mask(
            Gimp.ProcedureSensitivityMask.DRAWABLE
        )
        procedure.set_menu_label(MENU_LABEL)
        procedure.add_menu_path("<Image>/Filters/Development/")

        procedure.add_string_argument(
            "model_name",
            "Model",
            "Semantic segmentation model",
            MODEL_DPT,
            GObject.ParamFlags.READWRITE,
        )

        procedure.set_documentation(
            "Run semantic segmentation and add the result as a new layer.",
            "DPT-Hybrid and SegFormer-B0 use separate Python environments "
            "and their respective inference adapters.",
            PLUGIN_ID,
        )
        return procedure

    def run(self, procedure, run_mode, image, drawables, config, run_data):
        input_path = None
        output_path = None
        temp_dir = None

        try:
            model_name = config.get_property("model-name")

            if run_mode == Gimp.RunMode.INTERACTIVE:
                GimpUi.init("semsegnew.py")
                use_header_bar = Gtk.Settings.get_default().get_property(
                    "gtk-dialogs-use-header"
                )
                dialog = GimpUi.Dialog(
                    use_header_bar=use_header_bar,
                    title="Semantic Segmentation (Development)",
                )
                dialog.add_button("_Cancel", Gtk.ResponseType.CANCEL)
                dialog.add_button("_Run", Gtk.ResponseType.OK)

                grid = Gtk.Grid(
                    column_spacing=12,
                    row_spacing=8,
                    margin=12,
                )
                label = Gtk.Label(label="Model:")
                label.set_halign(Gtk.Align.START)

                model_combo = Gtk.ComboBoxText()
                model_combo.append_text(MODEL_DPT)
                model_combo.append_text(MODEL_SEGFORMER)
                model_combo.set_active(
                    1 if model_name == MODEL_SEGFORMER else 0
                )

                force_cpu_check = Gtk.CheckButton(
                    label="Force CPU (DPT-Hybrid only)"
                )
                force_cpu_check.set_active(False)

                segformer_gpu_check = Gtk.CheckButton(
                    label="Use GPU (SegFormer-B0)"
                )
                segformer_gpu_check.set_active(False)

                grid.attach(label, 0, 0, 1, 1)
                grid.attach(model_combo, 1, 0, 1, 1)
                grid.attach(force_cpu_check, 0, 1, 2, 1)
                grid.attach(segformer_gpu_check, 0, 2, 2, 1)
                dialog.get_content_area().add(grid)
                dialog.show_all()

                response = dialog.run()
                selected_model = model_combo.get_active_text()
                dialog.destroy()

                if response != Gtk.ResponseType.OK:
                    return procedure.new_return_values(
                        Gimp.PDBStatusType.CANCEL, GLib.Error()
                    )

                if selected_model not in (MODEL_DPT, MODEL_SEGFORMER):
                    Gimp.message("Please select a valid segmentation model.")
                    return procedure.new_return_values(
                        Gimp.PDBStatusType.CANCEL, GLib.Error()
                    )

                model_name = selected_model
                force_cpu = force_cpu_check.get_active()
                segformer_use_gpu = segformer_gpu_check.get_active()

            else:
                force_cpu = False
                segformer_use_gpu = False

            if not drawables:
                Gimp.message(
                    "Select an image layer before running segmentation."
                )
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CALLING_ERROR, None
                )

            if model_name == MODEL_DPT:
                adapter_file = DPT_ADAPTER_FILE
                if force_cpu:
                    python_exe = DPT_PYTHON_EXE
                    device = "cpu"
                else:
                    python_exe = DPT_GPU_PYTHON_EXE
                    device = "cuda"
                output_name = "dpt-hybrid-mask.png"
                layer_name = (
                    "DPT-Hybrid — Segmentation (CPU)"
                    if device == "cpu"
                    else "DPT-Hybrid — Segmentation (GPU)"
                )
            elif model_name == MODEL_SEGFORMER:
                adapter_file = SEGFORMER_ADAPTER_FILE
                if segformer_use_gpu:
                    python_exe = SEGFORMER_GPU_PYTHON_EXE
                    device = "cuda"
                else:
                    python_exe = SEGFORMER_PYTHON_EXE
                    device = "cpu"
                output_name = "segformer-mask.png"
                layer_name = (
                    "SegFormer-B0 — Segmentation (GPU)"
                    if device == "cuda"
                    else "SegFormer-B0 — Segmentation (CPU)"
                )
            else:
                Gimp.message(f"Unsupported model selection: {model_name}")
                return procedure.new_return_values(
                    Gimp.PDBStatusType.CALLING_ERROR, None
                )

            if not adapter_file.is_file():
                raise FileNotFoundError(
                    f"Adapter not found for {model_name}: {adapter_file}"
                )

            if not python_exe.is_file():
                raise FileNotFoundError(
                    f"Python interpreter not found for {model_name}: "
                    f"{python_exe}"
                )

            temp_dir = Path(tempfile.mkdtemp(prefix="gimp-semseg-"))
            input_path = temp_dir / "input.png"
            output_path = temp_dir / output_name

            export_proc = Gimp.get_pdb().lookup_procedure("file-png-export")
            if export_proc is None:
                raise RuntimeError("GIMP PNG export procedure is unavailable.")

            export_config = export_proc.create_config()
            export_config.set_property("run-mode", Gimp.RunMode.NONINTERACTIVE)
            export_config.set_property("image", image)
            export_config.set_property(
                "file", Gio.File.new_for_path(str(input_path))
            )

            export_result = export_proc.run(export_config)
            export_status = export_result.index(0)
            if export_status != Gimp.PDBStatusType.SUCCESS:
                raise RuntimeError(
                    f"PNG export failed with status {export_status}."
                )

            if not input_path.is_file():
                raise RuntimeError(
                    "PNG export returned success but input file is missing."
                )

            sys.path.insert(0, str(PROJECT_ROOT))

            if model_name == MODEL_DPT:
                from models.dpt_hybrid.adapter import run_inference
            else:
                from models.segformer_b0.adapter import run_inference

            inference_options = {
                "python_executable": str(python_exe),
            }
            if model_name in (MODEL_DPT, MODEL_SEGFORMER):
                inference_options["device"] = device

            run_inference(
                input_path,
                output_path,
                **inference_options,
            )

            if not output_path.is_file():
                raise RuntimeError(
                    f"{model_name} inference did not create an output PNG."
                )

            result_layer = Gimp.file_load_layer(
                Gimp.RunMode.NONINTERACTIVE,
                image,
                Gio.File.new_for_path(str(output_path)),
            )
            if result_layer is None:
                raise RuntimeError(
                    f"GIMP could not load the {model_name} segmentation PNG."
                )

            result_layer.set_name(layer_name)
            if not image.insert_layer(result_layer, None, 0):
                raise RuntimeError(
                    "GIMP could not insert the segmentation layer."
                )

            Gimp.message(
                f"{model_name} completed successfully. "
                "The segmentation mask was added as a new layer."
            )
            return procedure.new_return_values(
                Gimp.PDBStatusType.SUCCESS, None
            )

        except Exception as exc:
            traceback.print_exc()
            Gimp.message(f"Semantic Segmentation failed: {exc}")
            return procedure.new_return_values(
                Gimp.PDBStatusType.EXECUTION_ERROR, None
            )

        finally:
            for path in (input_path, output_path):
                try:
                    if path and path.is_file():
                        path.unlink()
                except OSError:
                    pass

            if temp_dir is not None:
                try:
                    temp_dir.rmdir()
                except OSError:
                    pass


Gimp.main(SemanticSegmentationPlugin.__gtype__, sys.argv)
