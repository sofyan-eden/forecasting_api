FROM ubuntu
WORKDIR /app
# COPY . .
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential cmake ca-certificates curl wget git \
    libcurl4-openssl-dev libgflags-dev unzip pkg-config \
    software-properties-common

RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcpu.zip && unzip libtorch-cxx11-abi-shared-with-deps-1.8.1+cpu.zip

RUN apt-get update

RUN add-apt-repository ppa:kip/pistache
RUN add-apt-repository ppa:pistache+team/unstable

RUN cmake .
RUN make 
ENTRYPOINT ["./server"]
CMD []