from setuptools import setup, find_packages

setup(
    name="StudentBehaviorMonitor",
    version="0.1.0",
    packages=find_packages(),
    author="yuta morimoto",
    author_email="s2222081@stu.musashino-u.ac.jp",
    description="StudentBehaviorMonitor",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yuta0726/StudentBehaviorMonitor",
    python_requires=">=3.12.3",
)