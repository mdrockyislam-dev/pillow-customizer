FROM node:20-bookworm-slim

ENV NODE_ENV=production \
    ONNXRUNTIME_NODE_INSTALL_CUDA=skip
WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY scripts ./scripts
RUN npm run check:native

COPY src ./src

EXPOSE 3000
CMD ["npm", "start"]
