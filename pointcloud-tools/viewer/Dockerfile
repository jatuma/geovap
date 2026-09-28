FROM node:22-alpine AS builder

ARG POTREE_REF=develop

RUN apk add --no-cache git
RUN git clone https://github.com/potree/potree.git /potree
WORKDIR /potree
RUN git checkout ${POTREE_REF}
# postinstall runs `gulp build pack`, producing build/potree/*
RUN npm install

# Guard: panos.py and consolidated/index.html are hand-tuned against a specific Images360
# implementation (sphere orientation formula, mirrored-texture default). If a newer Potree
# revision changes either, our viewer code silently breaks (wrong sphere yaw, flipped photos).
# Fail the build loudly instead so the mismatch gets noticed before it reaches production.
RUN grep -q "degToRad(-course + 90)" build/potree/potree.js \
    && grep -q "repeat.x = -1" build/potree/potree.js \
    || (echo "Potree API guard failed: degToRad(-course + 90) or repeat.x = -1 not found in build/potree/potree.js - Images360 implementation changed, review mapping/panos.py and consolidated/index.html" && exit 1)

FROM nginx:alpine

RUN mkdir -p /usr/share/nginx/html/potree
COPY --from=builder /potree/build /usr/share/nginx/html/potree/build
COPY --from=builder /potree/libs /usr/share/nginx/html/potree/libs

COPY index.html /usr/share/nginx/html/index.html
COPY view.html /usr/share/nginx/html/potree/view.html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# ./output (converted octrees) is bind-mounted here at runtime
RUN mkdir -p /usr/share/nginx/html/pointclouds
