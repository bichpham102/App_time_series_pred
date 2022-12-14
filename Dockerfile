FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120 \
    LC_ALL=C.UTF-8 \
    LANG=C.UTF-8

# we probably need build tools?
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    python3-dev \
    curl \
    wget \
    git

WORKDIR /app
COPY requirements.txt requirements.txt

# if we have a packages.txt, install it
# but packages.txt must have only LF endings
# COPY packages.txt packages.txt
# RUN xargs -a packages.txt apt-get install --yes

RUN pip install --no-cache-dir --upgrade -r requirements.txt

EXPOSE 8501

COPY . .

CMD ["streamlit", "run", "main.py"]

# docker build --progress=plain --tag canada:latest .
# docker run -ti -p 8501:8501 --rm canada:latest /bin/bash
# docker run -ti -p 8501:8501 --rm canada:latest
# docker run -ti -p 8501:8501 -v ${pwd}:/app --rm canada:latest
# docker run -ti -p 8501:8501 -v ${pwd}:/app --rm canada:latest /bin/bash
