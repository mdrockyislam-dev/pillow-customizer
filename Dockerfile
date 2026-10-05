FROM node:20-bookworm-slim

ENV NODE_ENV=production
ENV MALLOC_ARENA_MAX=2
ENV OMP_NUM_THREADS=1
ENV UV_THREADPOOL_SIZE=2

WORKDIR /app

COPY package*.json ./
RUN npm install --omit=dev

COPY src ./src

EXPOSE 3000

CMD ["npm", "start"]
