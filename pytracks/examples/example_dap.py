import data
import matplotlib.pyplot as plt
import tkinter as tk

from scipy.ndimage.morphology import binary_fill_holes
from skimage import io
from skimage.filters import threshold_isodata
from tkinter import messagebox


def example_dap():
    """
    """

    TITLE_WINDOWS = 'DAP example'

    # Step 1. Reading the input image.
    helping_window(TITLE_WINDOWS, 'Reading the input image.')
    image = data.dap01()
    plt.imshow(image, cmap='gray')
    plt.show()

    # Step 2. Binarizing and preprocessing it.
    helping_window(TITLE_WINDOWS, 'Binarizing and preprocessing the image.')
    thresh = threshold_isodata(image)
    image_bin = binary_fill_holes(thresh > image)
    plt.imshow(image_bin, cmap='gray')
    plt.show()

    # Step 3. Using the WUSEM segmentation.
    helping_window(TITLE_WINDOWS, 'Using WUSEM to separate overlapping tracks.')
    image_labels, _, _ = segmentation_wusem(image_bin,
                                            initial_radius=10,
                                            delta_radius=2)
    plt.imshow(image_labels, cmap='gray')
    plt.show()

    # Step 4. Clearing the right and lower borders.
    helping_window(TITLE_WINDOWS, 'Clearing tracks in right and lower borders.')
    image_labels = clear_rd_border(image_labels)
    plt.imshow(image_labels, cmap='gray')
    plt.show()

    # Step 5. Enumerating objects.
    helping_window(TITLE_WINDOWS, 'Enumerating objects found in the image.')
    img_numbers = enumerate_objects(image, img_labels, font_size=25)

    return None


def helping_window(title='default', message='default'):
    """Presents an info window on the screen.

    Parameters
    ----------
    title : string
        Title of the helping window.
    message : string
        Message the window will show.

    Notes
    -----
    Based on the Tk messagebox.showinfo() function.
    """

    window = tk.Tk()
    window.withdraw()

    messagebox.showinfo(title=title,
                        message=message)
    window.destroy()

    return None


example_dap()

