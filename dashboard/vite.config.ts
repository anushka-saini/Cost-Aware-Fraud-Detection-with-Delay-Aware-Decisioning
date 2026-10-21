import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'

// The dashboard calls the API through /api (HTTP and WebSocket), so the
// browser stays on one origin. Set FRAUD_API_URL if the API is elsewhere.
const target = process.env.FRAUD_API_URL || 'http://127.0.0.1:8000'
const proxy = {
  '/api': { target, changeOrigin: true, ws: true, rewrite: (path: string) => path.replace(/^\/api/, '') },
}

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) } },
  server: { port: 5173, host: true, proxy },
  preview: { port: 5173, host: true, proxy },
})
