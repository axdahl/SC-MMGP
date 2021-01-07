FROM tensorflow/tensorflow:1.3.0-gpu-py3

RUN apt-get update

COPY mmgp /experiments/mmgp/
COPY examples /experiments/examples/
COPY setup.py /experiments/setup.py
COPY datasets.zip /experiments/datasets.zip

WORKDIR /experiments

RUN unzip datasets.zip

RUN mkdir -p results

RUN pip install --upgrade pip

ENV PYTHONPATH="/experiments"

# suppress TF warnings
ENV TF_CPP_MIN_LOG_LEVEL=2

RUN python3 setup.py

CMD ["python", "examples/solar/p25_nonsparse_cmmgp.py"]
