import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.processor import process_images


def test_pipeline_execution():
    process_images()
    assert os.path.exists("outputs")
    assert len(os.listdir("outputs")) > 0
