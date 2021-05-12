FROM python:3
COPY /var/lib/jenkins/workspace/Question-Answer-Generator ./
WORKDIR ./
RUN pip install --no-cache-dir -r requirements.txt

