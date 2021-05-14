FROM python:3
COPY ./ ./
WORKDIR ./
RUN pip install -r requirements.txt
CMD ["python", "Final.py"]
