import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // The app imports model params, metrics and plots from the repo root
    // (../checkpoints, ../results), so the dev server must be allowed to read them
    fs: { allow: ['..'] },
  },
})
