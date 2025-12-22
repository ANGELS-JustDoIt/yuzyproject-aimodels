# app/clipboard/clipboard.py

import pyperclip


def copy_to_clipboard(text: str) -> None:
    """텍스트를 클립보드에 복사"""
    pyperclip.copy(text)


def paste_from_clipboard() -> str:
    """클립보드 텍스트 읽기"""
    return pyperclip.paste()
