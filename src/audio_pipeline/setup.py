from glob import glob

from setuptools import find_packages, setup

package_name = "audio_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Hardware interfacing nodes: VAD, STT, TTS",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vad_node = audio_pipeline.vad_node:main",
            "stt_node = audio_pipeline.stt_node:main",
            "tts_node = audio_pipeline.tts_node:main",
            "audio_loopback_node = audio_pipeline.audio_loopback_node:main",
        ],
    },
)
