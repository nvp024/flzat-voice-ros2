from glob import glob

from setuptools import find_packages, setup


package_name = "vlm_pipeline"

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
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (
            f"share/{package_name}/prompts/companion_robot_v1",
            glob("prompts/companion_robot_v1/*.txt"),
        ),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Standalone replaceable vision-language model action server",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vlm_node = vlm_pipeline.vlm_node:main",
            "vlm_test_client = vlm_pipeline.vlm_test_client:main",
        ],
    },
)
