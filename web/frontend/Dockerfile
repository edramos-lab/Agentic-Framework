FROM node:20-alpine AS build

WORKDIR /app
COPY web/frontend/package.json web/frontend/package-lock.json* ./
RUN npm install

COPY web/frontend/ ./
RUN npm run build

# --- Production stage: serve with nginx ---
FROM nginx:alpine

COPY --from=build /app/dist /usr/share/nginx/html
COPY web/frontend/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80
