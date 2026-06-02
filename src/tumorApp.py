"""Flet app for uploading, previewing, and classifying a single image."""

import logging

import flet as ft

from inference import classify_image


LOGGER = logging.getLogger(__name__)


class TumorTrackerApp:
    def __init__(self, page: ft.Page):
        self.page = page
        self.page.title = "Quantum Tumor Classifier"
        self.page.padding = 20
        self.selected_image_path = None

        self.placeholder = ft.Container(
            width=400,
            height=300,
            border_radius=10,
            bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.ON_SURFACE),
            border=ft.Border.all(2, ft.Colors.with_opacity(0.2, ft.Colors.ON_SURFACE)),
            content=ft.Column(
                [
                    ft.Icon(
                        ft.Icons.IMAGE_OUTLINED,
                        size=60,
                        color=ft.Colors.with_opacity(0.3, ft.Colors.ON_SURFACE),
                    ),
                    ft.Text(
                        "No image selected",
                        color=ft.Colors.with_opacity(0.4, ft.Colors.ON_SURFACE),
                    ),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                horizontal_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            visible=True,
        )

        self.selected_image = ft.Image(
            src="",
            width=400,
            height=300,
            fit=ft.BoxFit.CONTAIN,
            border_radius=10,
            visible=True,
        )

        self.status = ft.Text(
            "No image selected.",
            italic=True,
            color=ft.Colors.SECONDARY,
        )

        self.classify_btn = ft.ElevatedButton(
            "Classify",
            on_click=self.classification,
        )
        
        self.image_select_btn = ft.ElevatedButton(
            "Choose an image",
            icon=ft.Icons.UPLOAD_FILE,
            on_click=self._handle_image_pick,
        )

        self.image_output = ft.Image(
            src="",
            width=400,
            height=300,
            border_radius=10,
            fit=ft.BoxFit.CONTAIN,
            visible=False,
        )

        self.output_container = ft.Container(
            width=400,
            height=300,
        )

        self.placeholder_text = ft.Text(
            "No results to show.",
            color=ft.Colors.SECONDARY,
        )

        self.output_text = ft.Text(
            "",
            size=16,
            weight=ft.FontWeight.BOLD,
        )

        self.output_container_content = ft.Text(
            "No results to show.",
            color=ft.Colors.SECONDARY,
        )

        page.add(
            ft.Row(
                [
                    ft.Text("Quantum Tumor Classifier", size=24, weight=ft.FontWeight.BOLD),
                ],
                alignment=ft.MainAxisAlignment.CENTER,
                margin=20,
            ),
            ft.Row(
                [
                    ft.Column(
                        [
                            ft.Stack(
                                [
                                    self.selected_image,
                                    self.placeholder,
                                ]
                            ),
                            self.image_select_btn,
                            self.status,
                        ],
                        width=400,
                    ),
                    ft.Column(
                        [
                            ft.Text("Select an option", size=20, weight=ft.FontWeight.BOLD),
                            self.classify_btn,
                        ],
                        spacing=10,
                        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
                    ),
                    ft.Container(
                        width=400,
                        height=300,
                        bgcolor=ft.Colors.with_opacity(0.05, ft.Colors.WHITE),
                        border_radius=10,
                        border=ft.Border.all(
                            2, ft.Colors.with_opacity(0.2, ft.Colors.WHITE)
                        ),
                        content=self.output_container_content,
                        alignment=ft.Alignment.CENTER,
                    ),
                ],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        )

    async def _handle_image_pick(self, e: ft.Event[ft.Button]):
        files = await ft.FilePicker().pick_files(
            allow_multiple=False,
            file_type=ft.FilePickerFileType.IMAGE,
        )
        if not files:
            return

        self.selected_image_path = files[0].path
        self.selected_image.src = self.selected_image_path
        self.placeholder.visible = False
        self.status.value = "Image sélectionnée."
        self.update()

    def update(self):
        """Trigger a UI update on the page."""
        self.page.update()

    def classification(self, e=None):
        if not self.selected_image_path:
            self.output_container_content.value = "Veuillez choisir une image."
            self.update()
            return

    def classification(self):
        self.output_container_content.value = "Classification in progress..."
        self.update()
        pass
    

def main(page: ft.Page):
    """Launch the image uploader interface on a Flet page."""
    TumorTrackerApp(page)


if __name__ == "__main__":
    ft.run(main)