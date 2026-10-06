import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dashboard calls the FastAPI backend through /api so the browser stays
// on one origin and the API needs no CORS configuration.
const apiProxy = {
  '/api': {
    target: process.env.FRAUD_API_URL || 'http://127.0.0.1:8000',
    changeOrigin: true,
    rewrite: (path) => path.replace(/^\/api/, ''),
  },
}

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy: apiProxy },
  preview: { port: 5173, proxy: apiProxy },
})
