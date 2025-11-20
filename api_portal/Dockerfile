# Stage 1: Build
FROM node:24-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy the rest of the code
COPY . .

# Build the Next.js app
RUN npm run build

# Stage 2: Production image
FROM node:24-alpine AS runner

WORKDIR /app

# Copy built app from builder
COPY --from=builder /app/ ./

# Expose port
EXPOSE 3000

# Set environment
ENV NODE_ENV=production

# Start the app
CMD ["npm", "run", "start"]
