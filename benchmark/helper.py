import json
import platform
import re
import socket
import uuid

import cv2
import numpy as np
import psutil
from jinja2 import Template


def calculate_megapixels(image: np.ndarray):
    # Get the dimensions of the image
    height, width, _ = image.shape

    # Calculate the megapixels
    return (width * height) / 1e6


def resize_image_to_target_megapixels(
    image: np.ndarray, target_megapixels: float
) -> int:
    """
    Calculate megapixels and resize the image.

    Args:
        image: opencv image.
    """
    megapixels = calculate_megapixels(image)

    # Calculate the new width and height to achieve 0.5 megapixels
    if megapixels > target_megapixels:
        height, width, _ = image.shape
        aspect_ratio = width / height

        new_height = int((target_megapixels * 1e6 / aspect_ratio) ** 0.5)
        new_width = int(new_height * aspect_ratio)

        # Resize the image while maintaining the aspect ratio
        return cv2.resize(image, (new_width, new_height))

    # non-resized image will be return if it is below or equal to 0.5MP
    return image


def get_system_info():
    """Get system information."""

    try:
        info = {}
        info["platform"] = platform.system()
        info["platform-release"] = platform.release()
        info["platform-version"] = platform.version()
        info["architecture"] = platform.machine()
        info["hostname"] = socket.gethostname()
        info["ip-address"] = socket.gethostbyname(socket.gethostname())
        info["mac-address"] = ":".join(re.findall("..", "%012x" % uuid.getnode()))
        info["processor"] = platform.processor()
        info["ram"] = str(round(psutil.virtual_memory().total / (1024.0**3))) + " GB"
        return json.dumps(info)
    except Exception as e:
        print(e)


def generate_html_report(transaction_id, output_dir, metrics={}, images=[]):
    data = {
        "title": "Benchmark Report",
        "content": "This is a basic report generated by benchmark script.",
        "sections": [
            {
                "section_title": "System Information",
                "section_content": f"System information will be listed here. {get_system_info()}",
            },
            {
                "section_title": "Metrics",
                "section_content": f"Benchmark metrics will be listed here. {metrics}",
            },
            {
                "section_title": "Images",
                "section_content": f"Output images will be listed here. {images}",
            },
        ],
    }

    html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Benchmark Report</title>
        </head>
        <body>
            <h1>Benchmark Report</h1>
            <p>This is a basic report generated by benchmark script.</p>
            <h2>Sections</h2>
            <ul>
                <li>
                    <h3>System Information</h3>
                    <p>{get_system_info()}</p>
                </li>
                <li>
                    <h3>Metrics</h3>
                    <p>{metrics}</p>
                </li>
                <li>
                    <h3>Images</h3>
                    <p>{images}</p>
                </li>
            </ul>
        </body>
        </html>
    """

    template = Template(html_template)
    html_output = template.render(data=data)

    with open(f"{output_dir}/{transaction_id}_report.html", "w") as html_file:
        html_file.write(html_output)
