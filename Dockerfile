FROM python:3.6-slim
COPY ./ ./
WORKDIR ./
RUN pip install -r requirements.txt
CMD ["python", "Final.py"]
