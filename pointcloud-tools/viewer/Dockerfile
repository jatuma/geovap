FROM node:22-alpine AS builder

RUN apk add --no-cache git
RUN git clone --depth 1 https://github.com/potree/potree.git /potree
WORKDIR /potree
# postinstall runs `gulp build pack`, producing build/potree/*
RUN npm install

FROM nginx:alpine

RUN mkdir -p /usr/share/nginx/html/potree
COPY --from=builder /potree/build /usr/share/nginx/html/potree/build
COPY --from=builder /potree/libs /usr/share/nginx/html/potree/libs

COPY index.html /usr/share/nginx/html/index.html
COPY view.html /usr/share/nginx/html/potree/view.html
COPY nginx.conf /etc/nginx/conf.d/default.conf

# ./output (converted octrees) is bind-mounted here at runtime
RUN mkdir -p /usr/share/nginx/html/pointclouds
