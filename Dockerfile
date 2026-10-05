FROM node:20-bookworm-slim

ENV NODE_ENV=production
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV UV_THREADPOOL_SIZE=2

WORKDIR /app

COPY package*.json ./
RUN npm ci --omit=dev

COPY src ./src
COPY scripts ./scripts
COPY models ./models
RUN npm run check:native
RUN npm run cache:subject

EXPOSE 3000

CMD ["npm", "start"]
