FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        clang-17 \
        coreutils \
        python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/fctool

COPY fcbenchtool /usr/local/bin/fcbenchtool
COPY fcmultibench /usr/local/bin/fcmultibench
COPY bencharch_header.cpp /usr/local/share/fctool/bencharch_header.cpp
COPY bencharch_footer.cpp /usr/local/share/fctool/bencharch_footer.cpp

RUN chmod +x /usr/local/bin/fcbenchtool /usr/local/bin/fcmultibench

ENV CXX=clang++-17
WORKDIR /work

ENTRYPOINT ["fcmultibench"]
