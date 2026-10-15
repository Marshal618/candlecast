import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In dev, proxy API calls to the FastAPI backend so the frontend uses same-origin /api paths
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': `http://127.0.0.1:${process.env.API_PORT ?? 8010}`,
    },
  },
})
