FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libtbb-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# --recursive: laszip and brotli are vendored as git submodules
RUN git clone --recursive --depth 1 \
    https://github.com/potree/PotreeConverter.git /src

WORKDIR /src/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release \
    && cmake --build . --config Release -j "$(nproc)"

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libtbb12 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/build/PotreeConverter /usr/local/bin/PotreeConverter
COPY --from=builder /src/build/liblaszip.so* /usr/local/lib/
COPY --from=builder /src/build/resources /usr/local/share/potreeconverter/resources
RUN ldconfig

WORKDIR /data
ENTRYPOINT ["/usr/local/bin/PotreeConverter"]
