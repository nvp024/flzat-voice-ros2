from setuptools import setup, find_packages

package_name = "trigger_engine"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="your@email.com",
    description="Orchestrator / brain node: AudioVisualTrigger",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "audio_visual_trigger = trigger_engine.audio_visual_trigger:main",
        ],
    },
)
