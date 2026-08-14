from setuptools import find_packages, setup


package_name = "vision_pipeline"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        (
            "share/ament_index/resource_index/packages",
            [f"resource/{package_name}"],
        ),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", ["launch/vision_pipeline.launch.py"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Non-blocking camera capture, frame buffering, and motion events",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "camera_node = vision_pipeline.camera_node:main",
        ],
    },
)
