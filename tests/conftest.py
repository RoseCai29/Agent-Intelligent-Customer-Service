"""确保 tests 运行时项目根目录在 sys.path 上（使 `import app` 生效）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))