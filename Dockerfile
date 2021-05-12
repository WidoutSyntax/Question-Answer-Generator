FROM python:3
COPY ./ ./
WORKDIR ./
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "Final.py"]
