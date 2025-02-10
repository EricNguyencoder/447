FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set PATH for Rust
ENV PATH="/root/.cargo/bin:${PATH}"
RUN mkdir /job
WORKDIR /job
VOLUME ["/job/data", "/job/src", "/job/work", "/job/output"]

# You should install any dependencies you need here.
# RUN pip install tqdm
COPY requirements.txt /job/ 
RUN pip install transformers datasets accelerate peft bitsandbytes