import tkinter as tk

from tkinter import messagebox


def example_dap():
    """
    """

    helping_window('Step 1', 'Binarizing the input images.')

    return None


def helping_window(title='default', message='default'):
    """
    """

    window = tk.Tk()
    window.withdraw()

    messagebox.showinfo(title=title,
                        message=message)
    window.destroy()


    return None


example_dap()

