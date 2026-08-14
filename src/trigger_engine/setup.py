from glob import glob

from setuptools import find_packages, setup


package_name = "trigger_engine"

setup(
    name=package_name,
    version="0.2.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Bounded audio/visual fusion and asynchronous VLM/TTS manager",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "multimodal_manager = trigger_engine.multimodal_manager:main",
        ],
    },
)
