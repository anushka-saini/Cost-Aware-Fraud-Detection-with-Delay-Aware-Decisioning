# Build the dashboard, then serve it with Vite's preview server, which also
# proxies /api (HTTP and WebSocket) to the API container.
FROM node:22-alpine
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
RUN npm run build
ENV FRAUD_API_URL=http://api:8000
EXPOSE 5173
CMD ["npx", "vite", "preview", "--host", "0.0.0.0", "--port", "5173"]
