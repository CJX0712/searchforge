# SearchForge · 作者 晨星 (CJX0712)
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0

WORKDIR /app

COPY requirements.lock.txt ./
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r requirements.lock.txt

COPY . .

# 非 root 运行
RUN useradd -m -u 1000 forge && chown -R forge:forge /app
USER forge

# 冒烟：进入容器即验证可运行
CMD ["python", "examples/run_demo.py"]
