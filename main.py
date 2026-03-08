"""
AnnotFlow エントリポイント

QApplication を初期化し、メインウィンドウを表示する。
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> None:
    """アプリケーションを起動する。"""
    app = QApplication(sys.argv)
    app.setApplicationName("AnnotFlow")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.resize(1200, 800)
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
