from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="personavoice",
    version="10.4.8",
    description="PersonaVoice: Plug-in Adapter Architecture for 1-Second Extreme Voice Cloning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/personavoice/personavoice",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "speechbrain>=0.8.0",
        "transformers>=4.35.0",
        "accelerate>=0.24.0",
        "numpy>=1.24.0",
        "librosa>=0.10.0",
        "soundfile>=0.12.0",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "pyyaml>=6.0",
        "omegaconf>=2.3.0",
        "einops>=0.7.0",
        "tqdm>=4.66.0",
        "silero-vad>=5.0",
        "f5-tts>=1.0",
        "fastapi>=0.104.0",
        "uvicorn>=0.24.0",
        "imageio-ffmpeg>=0.4.9",
        "Pillow>=10.0.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
    ],
    keywords=(
        "voice-cloning, zero-shot-tts, few-shot-voice-cloning, flow-matching, "
        "F5-TTS, speech-synthesis, persona-driven-tts, emotional-tts, "
        "plug-in-adapter, 1-second-cloning, speaker-embedding, ECAPA-TDNN, "
        "orthogonal-decomposition, length-adaptive-generation, FiLM-modulation"
    ),
)
